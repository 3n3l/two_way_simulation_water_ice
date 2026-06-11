from taichi.linalg import SparseMatrixBuilder, SparseCG

import taichi as ti


@ti.data_oriented
class HeatSolver:
    def __init__(self, solver) -> None:
        self.w_cells = solver.wx * solver.wy * solver.wz
        self.solver = solver

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        for i, j, k in ti.ndrange(self.solver.wx, self.solver.wy, self.solver.wz):
            idx = i + self.solver.wx * (j + (self.solver.wy * k))  # raveled index, 3D
            b[idx] = self.solver.temperature_c[i, j, k]  # right-hand side

            # We enforce Dirichlet temperature boundary conditions at CELLS that are in contact with fixed
            # temperature bodies (like a heated pan (-> boundary cells in our case) or air (-> empty cells)),
            # i.e we keep the currently recorded cell temperatures for empty (air) cells.
            if not self.solver.is_interior(i, j, k):
                A[idx, idx] += 1.0
                continue

            # Compute (1 / dx^2) * ((dt * dx^d) / (m_c * c_c)) [Jiang 2016, Ch. 5.8],
            dt_inv_mass_capacity = self.solver.dt[None] * self.solver.dx
            dt_inv_mass_capacity /= self.solver.mass_c[i, j, k] * self.solver.capacity_c[i, j, k]
            inv_dx_sqrd = self.solver.inv_dx * self.solver.inv_dx
            diagonal = 1.0  # to keep max_num_triplets as low as possible

            # We enforce homogeneous Neumann boundary conditions at FACES adjacent to cells that are corresponding
            # to insulated objects, i.e we set the conductivity to zero for faces adjacent to insulated cells
            # (or by simply just not incorporating them).
            if not self.solver.is_insulated(i + 1, j, k):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i + 1, j, k]
                if self.solver.is_empty(i + 1, j, k):  # non-homogeneous Dirichlet
                    c = self.solver.conductivity_x[i + 1, j, k]
                    A[idx, idx + 1] -= dt_inv_mass_capacity * c
                    b[idx] += inv_dx_sqrd * c * self.solver.temperature_c[i + 1, j, k]

            if not self.solver.is_insulated(i - 1, j, k):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_x[i, j, k]
                if self.solver.is_empty(i - 1, j, k):  # non-homogeneous Dirichlet
                    c = self.solver.conductivity_x[i, j, k]
                    A[idx, idx - 1] -= dt_inv_mass_capacity * c
                    b[idx] += inv_dx_sqrd * c * self.solver.temperature_c[i - 1, j, k]

            if not self.solver.is_insulated(i, j + 1, k):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j + 1, k]
                if self.solver.is_empty(i, j + 1, k):  # non-homogeneous Dirichlet
                    c = self.solver.conductivity_y[i, j + 1, k]
                    A[idx, idx + self.solver.wx] -= dt_inv_mass_capacity * c
                    b[idx] += inv_dx_sqrd * c * self.solver.temperature_c[i, j + 1, k]

            if not self.solver.is_insulated(i, j - 1, k):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_y[i, j, k]
                if self.solver.is_empty(i, j - 1, k):  # non-homogeneous Dirichlet
                    c = self.solver.conductivity_y[i, j, k]
                    A[idx, idx - self.solver.wx] -= dt_inv_mass_capacity * c
                    b[idx] += inv_dx_sqrd * c * self.solver.temperature_c[i, j - 1, k]

            if not self.solver.is_insulated(i, j, k + 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_z[i, j, k + 1]
                if self.solver.is_empty(i, j, k + 1):  # non-homogeneous Dirichlet
                    c = self.solver.conductivity_z[i, j, k + 1]
                    A[idx, idx + (self.solver.wx * self.solver.wz)] -= dt_inv_mass_capacity * c
                    b[idx] += inv_dx_sqrd * c * self.solver.temperature_c[i, j, k + 1]

            if not self.solver.is_insulated(i, j, k - 1):  # homogeneous Neumann
                diagonal += dt_inv_mass_capacity * self.solver.conductivity_z[i, j, k]
                if self.solver.is_empty(i, j, k - 1):  # non-homogeneous Dirichlet
                    c = self.solver.conductivity_z[i, j, k]
                    A[idx, idx - (self.solver.wx * self.solver.wz)] -= dt_inv_mass_capacity * c
                    b[idx] += inv_dx_sqrd * c * self.solver.temperature_c[i, j, k - 1]

            A[idx, idx] += diagonal  # add value from variable, to keep max_num_triplets as low as possible

    @ti.kernel
    def fill_temperature_field(self, T: ti.types.ndarray()):  # pyright: ignore
        for i, j, k in ti.ndrange(self.solver.wx, self.solver.wy, self.solver.wz):
            idx = i + self.solver.wx * (j + (self.solver.wy * k))  # raveled index, 3D
            self.solver.temperature_c[i, j, k] = T[idx]

    def solve(self):
        A = SparseMatrixBuilder(
            max_num_triplets=(7 * self.w_cells),
            num_rows=self.w_cells,
            num_cols=self.w_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.w_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system.
        solver = SparseCG(A.build(), b, atol=1e-5, max_iter=500)
        T, _ = solver.solve()

        self.fill_temperature_field(T)
