#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- シミュレーション・パラメータ ---
#define NMOL 200             // 分子数
#define SITES_PER_MOL 4      // 1分子あたりのサイト数 (ブタンのUAモデル: CH3-CH2-CH2-CH3)
#define NATOMS (NMOL * SITES_PER_MOL) // 全原子数

#define BOX_SIZE 100.0        // シミュレーションボックスの1辺の長さ (Angstrom)
#define CUTOFF 13.65          // LJカットオフ距離
#define CUTOFF2 (CUTOFF * CUTOFF)

#define DT 0.002             // タイムステップ (ps)
#define NSTEPS 1000          // 総ステップ数
#define MASS 14.0            // CH2/CH3の近似質量 (g/mol)

// --- 力場パラメータ ---
#define EPSILON 0.1          // kcal/mol
#define SIGMA 3.9            // Angstrom
#define K_BOND 500.0         // 結合力のバネ定数 (kcal/mol/A^2)
#define R0_BOND 1.54         // 平衡結合長 (Angstrom)

// 結合角と二面角のパラメータ
#define K_ANGLE 62.1         // 結合角のバネ定数 (kcal/mol/rad^2)
#define THETA0  1.99         // 平衡結合角 (114度 = 約 1.99 rad)
#define V1  1.411            // 二面角パラメータ1 (kcal/mol)
#define V2 -0.271            // 二面角パラメータ2 (kcal/mol)
#define V3  3.145            // 二面角パラメータ3 (kcal/mol)

// --- 単位変換とエネルギー計算用定数 ---
#define EC_FAC 418.4          // kcal/mol -> (g/mol)*(A/ps)^2 の変換係数
#define R_GAS 0.0019872       // 気体定数 (kcal/mol/K)

// --- データ構造 ---
typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
    int mol_id;              // 所属する分子のID (0 to NMOL-1)
} Particle;

Particle atoms[NATOMS];

// --- エネルギー・温度記録用変数 ---
double ep_lj = 0.0, ep_bond = 0.0, ep_angle = 0.0, ep_dih = 0.0;
double e_kin = 0.0, e_pot = 0.0, e_tot = 0.0, temperature = 0.0;

// --- acos近似用変数 ---
double COS_THETA0;
double acos_coeffs[10];

// ★ 追加: 誤差確認用のグローバル変数
double mol0_exact_theta1_deg = 0.0;
double mol0_approx_theta1_deg = 0.0;
double avg_error_deg = 0.0;

// --- 関数の宣言 ---
void init_acos_poly();
void init_system();
void apply_pbc(double *dx, double *dy, double *dz);
void compute_forces();
void compute_lj();
void compute_bonds();
void compute_angles();
void compute_dihedrals();
void integrate_step1();
void integrate_step2();
void compute_energy_and_temp();
void cross_product(double ax, double ay, double az, double bx, double by, double bz, double *cx, double *cy, double *cz);
double dot_product(double ax, double ay, double az, double bx, double by, double bz);

// --- メイン関数 ---
int main() {
    // 最初にacosの近似多項式の係数を計算
    init_acos_poly();
    
    init_system();
    
    // 初期の力を計算して初回のエネルギーを出力
    compute_forces();
    compute_energy_and_temp();
    printf("Initial | T: %6.1f K | PE: %8.1f | KE: %8.1f | Tot: %8.1f\n", 
           temperature, e_pot, e_kin, e_tot);
    
    for (int step = 1; step <= NSTEPS; step++) {
        integrate_step1(); 
        compute_forces();  
        integrate_step2(); 
        
        // ★ 追加: 毎ステップ、分子0の結合角（厳密値・近似値）と系全体の平均誤差を出力
        printf("Step: %5d | Exact: %8.4f deg | Approx: %8.4f deg | Avg Error: %.4e deg\n",
               step, mol0_exact_theta1_deg, mol0_approx_theta1_deg, avg_error_deg);

        if (step % 100 == 0) {
            compute_energy_and_temp();
            printf("Step: %5d | T: %6.1f K | PE: %8.1f | KE: %8.1f | Tot: %8.1f\n", 
                   step, temperature, e_pot, e_kin, e_tot);
        }
    }
    
    printf("Simulation finished.\n");
    return 0;
}

// --- acosの9次多項式近似係数の初期化 ---
void init_acos_poly() {
    COS_THETA0 = cos(THETA0);
    double x0 = COS_THETA0;
    double sin_th0 = sin(THETA0);
    double w = 1.0 - x0 * x0; 

    acos_coeffs[0] = THETA0;
    acos_coeffs[1] = -1.0 / sin_th0;
    acos_coeffs[2] = (x0 * acos_coeffs[1]) / (2.0 * w);

    for (int k = 3; k <= 9; k++) {
        int n = k - 2;
        double term1 = ((2.0 * n + 1.0) / (n + 2.0)) * x0 * acos_coeffs[k-1];
        double term2 = ((double)(n * n) / ((n + 1.0) * (n + 2.0))) * acos_coeffs[k-2];
        acos_coeffs[k] = (term1 + term2) / w;
    }
}

// --- システムの初期化 ---
void init_system() {
    int idx = 0;
    int n_per_side = ceil(cbrt(NMOL));
    double spacing = BOX_SIZE / n_per_side;
    
    // ジグザグ配置用の座標計算用パラメータ
    double half_theta = THETA0 / 2.0;
    double dx_step = R0_BOND * sin(half_theta);
    double dy_step = R0_BOND * cos(half_theta);

    for (int i = 0; i < n_per_side && idx < NMOL; i++) {
        for (int j = 0; j < n_per_side && idx < NMOL; j++) {
            for (int k = 0; k < n_per_side && idx < NMOL; k++) {
                double base_x = i * spacing;
                double base_y = j * spacing;
                double base_z = k * spacing;
                
                for (int s = 0; s < SITES_PER_MOL; s++) {
                    int atom_idx = idx * SITES_PER_MOL + s;
                    
                    // ★ 修正: ジグザグ配置 (トランス配座) の生成
                    atoms[atom_idx].x = base_x + s * dx_step;
                    atoms[atom_idx].y = base_y + (s % 2 == 0 ? 0.0 : -dy_step);
                    atoms[atom_idx].z = base_z;
                    
                    atoms[atom_idx].vx = atoms[atom_idx].vy = atoms[atom_idx].vz = 0.0;
                    atoms[atom_idx].fx = atoms[atom_idx].fy = atoms[atom_idx].fz = 0.0;
                    atoms[atom_idx].mol_id = idx;
                }
                idx++;
            }
        }
    }
}

// --- 最小イメージ規約（PBC） ---
void apply_pbc(double *dx, double *dy, double *dz) {
    if (*dx >  BOX_SIZE / 2.0) *dx -= BOX_SIZE;
    if (*dx <= -BOX_SIZE / 2.0) *dx += BOX_SIZE;
    if (*dy >  BOX_SIZE / 2.0) *dy -= BOX_SIZE;
    if (*dy <= -BOX_SIZE / 2.0) *dy += BOX_SIZE;
    if (*dz >  BOX_SIZE / 2.0) *dz -= BOX_SIZE;
    if (*dz <= -BOX_SIZE / 2.0) *dz += BOX_SIZE;
}

// --- ベクトル演算の補助関数 ---
void cross_product(double ax, double ay, double az, 
                   double bx, double by, double bz, 
                   double *cx, double *cy, double *cz) {
    *cx = ay * bz - az * by;
    *cy = az * bx - ax * bz;
    *cz = ax * by - ay * bx;
}

double dot_product(double ax, double ay, double az, 
                   double bx, double by, double bz) {
    return ax * bx + ay * by + az * bz;
}

// =========================================================
// 力の計算ルーチン群
// =========================================================

void compute_forces() {
    for (int i = 0; i < NATOMS; i++) {
        atoms[i].fx = atoms[i].fy = atoms[i].fz = 0.0;
    }
    
    compute_lj();
    compute_bonds();
    compute_angles();
    compute_dihedrals();
}

void compute_lj() {
    ep_lj = 0.0;
    double sig6 = pow(SIGMA, 6);
    double sig12 = sig6 * sig6;
    double eps24 = 24.0 * EPSILON;
    
    for (int i = 0; i < NATOMS - 1; i++) {
        for (int j = i + 1; j < NATOMS; j++) {
            if (atoms[i].mol_id != atoms[j].mol_id) {
                double dx = atoms[i].x - atoms[j].x;
                double dy = atoms[i].y - atoms[j].y;
                double dz = atoms[i].z - atoms[j].z;
                apply_pbc(&dx, &dy, &dz);
                
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < CUTOFF2) {
                    double r2inv = 1.0 / r2;
                    double r6inv = r2inv * r2inv * r2inv;
                    double force_mag = eps24 * r6inv * r2inv * (2.0 * sig12 * r6inv - sig6);
                    
                    double term6 = sig6 * r6inv;
                    ep_lj += 4.0 * EPSILON * (term6 * term6 - term6);
                    
                    double fx = force_mag * dx;
                    double fy = force_mag * dy;
                    double fz = force_mag * dz;
                    
                    atoms[i].fx += fx; atoms[i].fy += fy; atoms[i].fz += fz;
                    atoms[j].fx -= fx; atoms[j].fy -= fy; atoms[j].fz -= fz;
                }
            }
        }
    }
}

void compute_bonds() {
    ep_bond = 0.0;
    for (int m = 0; m < NMOL; m++) {
        int base = m * SITES_PER_MOL;
        for (int i = 0; i < SITES_PER_MOL - 1; i++) {
            int a1 = base + i;
            int a2 = base + i + 1;
            
            double dx = atoms[a1].x - atoms[a2].x;
            double dy = atoms[a1].y - atoms[a2].y;
            double dz = atoms[a1].z - atoms[a2].z;
            apply_pbc(&dx, &dy, &dz);
            
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            double force_mag = -K_BOND * (r - R0_BOND); 
            
            ep_bond += 0.5 * K_BOND * (r - R0_BOND) * (r - R0_BOND);
            
            double fx = force_mag * (dx / r);
            double fy = force_mag * (dy / r);
            double fz = force_mag * (dz / r);
            
            atoms[a1].fx += fx; atoms[a1].fy += fy; atoms[a1].fz += fz;
            atoms[a2].fx -= fx; atoms[a2].fy -= fy; atoms[a2].fz -= fz;
        }
    }
}

// --- 分子内: 結合角 (Angle) ---
void compute_angles() {
    ep_angle = 0.0;
    
    // ★ 追加: 誤差計算用の一時変数
    double sum_error_local = 0.0;
    int count_angles_local = 0;

    for (int m = 0; m < NMOL; m++) {
        int base = m * SITES_PER_MOL;
        for (int i = 0; i < SITES_PER_MOL - 2; i++) {
            int a1 = base + i;
            int a2 = base + i + 1; // 中心原子
            int a3 = base + i + 2;

            double r12x = atoms[a1].x - atoms[a2].x;
            double r12y = atoms[a1].y - atoms[a2].y;
            double r12z = atoms[a1].z - atoms[a2].z;
            apply_pbc(&r12x, &r12y, &r12z);

            double r32x = atoms[a3].x - atoms[a2].x;
            double r32y = atoms[a3].y - atoms[a2].y;
            double r32z = atoms[a3].z - atoms[a2].z;
            apply_pbc(&r32x, &r32y, &r32z);

            double len_r12 = sqrt(r12x*r12x + r12y*r12y + r12z*r12z);
            double len_r32 = sqrt(r32x*r32x + r32y*r32y + r32z*r32z);
            
            double cos_theta = dot_product(r12x, r12y, r12z, r32x, r32y, r32z) / (len_r12 * len_r32);
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;
            
            // --- acosを 114° = THETA0 周りの9次多項式で近似 ---
            double dx = cos_theta - COS_THETA0;
            double approx_theta = acos_coeffs[0] + dx * (acos_coeffs[1] + dx * (acos_coeffs[2] + dx * (acos_coeffs[3] + 
                                  dx * (acos_coeffs[4] + dx * (acos_coeffs[5] + dx * (acos_coeffs[6] + dx * (acos_coeffs[7] + 
                                  dx * (acos_coeffs[8] + dx * acos_coeffs[9]))))))));

            // ★ 追加: 厳密なacosを計算して誤差を取得
            double exact_theta = acos(cos_theta);
            
            sum_error_local += fabs(exact_theta - approx_theta);
            count_angles_local++;

            // ★ 追加: 分子0の1つ目の角度を記録
            if (m == 0 && i == 0) {
                mol0_exact_theta1_deg = exact_theta * (180.0 / M_PI);
                mol0_approx_theta1_deg = approx_theta * (180.0 / M_PI);
            }

            // 本計算には近似値(approx_theta)を使用する
            double theta = approx_theta;

            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            double dV_dtheta = K_ANGLE * (theta - THETA0); 
            
            ep_angle += 0.5 * K_ANGLE * (theta - THETA0) * (theta - THETA0);
            
            double coef = (sin_theta > 1e-5) ? (-dV_dtheta / sin_theta) : 0.0;

            double f1x = coef * (r32x / (len_r12 * len_r32) - cos_theta * r12x / (len_r12 * len_r12));
            double f1y = coef * (r32y / (len_r12 * len_r32) - cos_theta * r12y / (len_r12 * len_r12));
            double f1z = coef * (r32z / (len_r12 * len_r32) - cos_theta * r12z / (len_r12 * len_r12));

            double f3x = coef * (r12x / (len_r12 * len_r32) - cos_theta * r32x / (len_r32 * len_r32));
            double f3y = coef * (r12y / (len_r12 * len_r32) - cos_theta * r32y / (len_r32 * len_r32));
            double f3z = coef * (r12z / (len_r12 * len_r32) - cos_theta * r32z / (len_r32 * len_r32));

            atoms[a1].fx += f1x; atoms[a1].fy += f1y; atoms[a1].fz += f1z;
            atoms[a3].fx += f3x; atoms[a3].fy += f3y; atoms[a3].fz += f3z;
            atoms[a2].fx -= (f1x + f3x); 
            atoms[a2].fy -= (f1y + f3y); 
            atoms[a2].fz -= (f1z + f3z);
        }
    }

    // ★ 追加: 全体の平均誤差（度）を計算してグローバル変数へ
    if (count_angles_local > 0) {
        avg_error_deg = (sum_error_local / count_angles_local) * (180.0 / M_PI);
    }
}

// --- 分子内: 二面角 (Dihedral / Torsion) ---
void compute_dihedrals() {
    ep_dih = 0.0;
    for (int m = 0; m < NMOL; m++) {
        int base = m * SITES_PER_MOL;
        for (int i = 0; i < SITES_PER_MOL - 3; i++) {
            int a1 = base + i;
            int a2 = base + i + 1;
            int a3 = base + i + 2;
            int a4 = base + i + 3;

            double r12x = atoms[a2].x - atoms[a1].x;
            double r12y = atoms[a2].y - atoms[a1].y;
            double r12z = atoms[a2].z - atoms[a1].z;
            apply_pbc(&r12x, &r12y, &r12z);

            double r23x = atoms[a3].x - atoms[a2].x;
            double r23y = atoms[a3].y - atoms[a2].y;
            double r23z = atoms[a3].z - atoms[a2].z;
            apply_pbc(&r23x, &r23y, &r23z);

            double r34x = atoms[a4].x - atoms[a3].x;
            double r34y = atoms[a4].y - atoms[a3].y;
            double r34z = atoms[a4].z - atoms[a3].z;
            apply_pbc(&r34x, &r34y, &r34z);

            double mx, my, mz, nx, ny, nz;
            cross_product(r12x, r12y, r12z, r23x, r23y, r23z, &mx, &my, &mz);
            cross_product(r23x, r23y, r23z, r34x, r34y, r34z, &nx, &ny, &nz);

            double m_sq = mx*mx + my*my + mz*mz;
            double n_sq = nx*nx + ny*ny + nz*nz;
            double len_r23 = sqrt(r23x*r23x + r23y*r23y + r23z*r23z);

            if (m_sq >= 1e-10 && n_sq >= 1e-10) {
                double cos_phi = dot_product(mx, my, mz, nx, ny, nz) / sqrt(m_sq * n_sq);
                if (cos_phi > 1.0) cos_phi = 1.0;
                if (cos_phi < -1.0) cos_phi = -1.0;

                double phi = acos(cos_phi);
                double sign_val = dot_product(r12x, r12y, r12z, nx, ny, nz);
                if (sign_val < 0.0) phi = -phi;

                double sin_phi = sin(phi);
                double sin_2phi = sin(2.0 * phi);
                double sin_3phi = sin(3.0 * phi);
                
                double cos_2phi = cos(2.0 * phi);
                double cos_3phi = cos(3.0 * phi);
                
                double dV_dphi = -0.5 * V1 * sin_phi + V2 * sin_2phi - 1.5 * V3 * sin_3phi;

                ep_dih += 0.5 * (V1*(1.0 + cos_phi) + V2*(1.0 - cos_2phi) + V3*(1.0 + cos_3phi));

                double f1_coef = dV_dphi * len_r23 / m_sq;
                double f4_coef = dV_dphi * len_r23 / n_sq;

                double f1x = f1_coef * mx;
                double f1y = f1_coef * my;
                double f1z = f1_coef * mz;

                double f4x = -f4_coef * nx;
                double f4y = -f4_coef * ny;
                double f4z = -f4_coef * nz;

                double coef2 = dot_product(r12x, r12y, r12z, r23x, r23y, r23z) / (len_r23 * len_r23);
                double coef3 = dot_product(r34x, r34y, r34z, r23x, r23y, r23z) / (len_r23 * len_r23);

                double f2x = (coef2 - 1.0) * f1x - coef3 * f4x;
                double f2y = (coef2 - 1.0) * f1y - coef3 * f4y;
                double f2z = (coef2 - 1.0) * f1z - coef3 * f4z;

                double f3x = -coef2 * f1x + (coef3 - 1.0) * f4x;
                double f3y = -coef2 * f1y + (coef3 - 1.0) * f4y;
                double f3z = -coef2 * f1z + (coef3 - 1.0) * f4z;

                atoms[a1].fx += f1x; atoms[a1].fy += f1y; atoms[a1].fz += f1z;
                atoms[a2].fx += f2x; atoms[a2].fy += f2y; atoms[a2].fz += f2z;
                atoms[a3].fx += f3x; atoms[a3].fy += f3y; atoms[a3].fz += f3z;
                atoms[a4].fx += f4x; atoms[a4].fy += f4y; atoms[a4].fz += f4z;
            }
        }
    }
}

// =========================================================
// その他機能
// =========================================================

void compute_energy_and_temp() {
    e_kin = 0.0;
    for (int i = 0; i < NATOMS; i++) {
        double v2 = atoms[i].vx * atoms[i].vx + 
                    atoms[i].vy * atoms[i].vy + 
                    atoms[i].vz * atoms[i].vz;
        e_kin += 0.5 * MASS * v2;
    }
    e_kin /= EC_FAC; 
    
    e_pot = ep_lj + ep_bond + ep_angle + ep_dih;
    e_tot = e_kin + e_pot; 
    
    temperature = (2.0 * e_kin) / (3.0 * NATOMS * R_GAS);
}

void integrate_step1() {
    double dt_half_mass = ((DT * 0.5) / MASS) * EC_FAC;
    for (int i = 0; i < NATOMS; i++) {
        atoms[i].vx += atoms[i].fx * dt_half_mass;
        atoms[i].vy += atoms[i].fy * dt_half_mass;
        atoms[i].vz += atoms[i].fz * dt_half_mass;
        
        atoms[i].x += atoms[i].vx * DT;
        atoms[i].y += atoms[i].vy * DT;
        atoms[i].z += atoms[i].vz * DT;
        
        if(atoms[i].x < 0) atoms[i].x += BOX_SIZE;
        if(atoms[i].x >= BOX_SIZE) atoms[i].x -= BOX_SIZE;
        if(atoms[i].y < 0) atoms[i].y += BOX_SIZE;
        if(atoms[i].y >= BOX_SIZE) atoms[i].y -= BOX_SIZE;
        if(atoms[i].z < 0) atoms[i].z += BOX_SIZE;
        if(atoms[i].z >= BOX_SIZE) atoms[i].z -= BOX_SIZE;
    }
}

void integrate_step2() {
    double dt_half_mass = ((DT * 0.5) / MASS) * EC_FAC;
    for (int i = 0; i < NATOMS; i++) {
        atoms[i].vx += atoms[i].fx * dt_half_mass;
        atoms[i].vy += atoms[i].fy * dt_half_mass;
        atoms[i].vz += atoms[i].fz * dt_half_mass;
    }
}
