from src.solvers.staggered_solver import StaggeredSolver

import taichi as ti


@ti.data_oriented
class PressureSolver_Jacobi:
    def __init__(self, solver: StaggeredSolver) -> None:
        self.solver = solver

        self.jacobi_iter = 100
        self.jacobi_weight = 1.0
        self.rho = 1.0  # TODO: change to volume based rho

        self.grid_shape = (self.solver.wx, self.solver.wy, self.solver.wz)
        self.divergence = ti.field(ti.f32, shape=self.grid_shape)
        self.pressure = ti.field(ti.f32, shape=self.grid_shape)
        self.pressure_new = ti.field(ti.f32, shape=self.grid_shape)

    @ti.kernel
    def compute_divergence(self):
        for i, j, k in self.divergence:
            div = 0.0
            if not self.solver.is_colliding(i, j, k):
                div += self.solver.velocity_x[i + 1, j, k] - self.solver.velocity_x[i, j, k]
                div += self.solver.velocity_y[i, j + 1, k] - self.solver.velocity_y[i, j, k]
                div += self.solver.velocity_z[i, j, k + 1] - self.solver.velocity_z[i, j, k]

            self.divergence[i, j, k] = div / self.solver.dx

    @ti.kernel
    def jacobi_iteration(self):
        for i, j, k in self.pressure:
            pressure = 0.0
            if self.solver.is_interior(i, j, k):
                div = self.divergence[i, j, k]

                p_x1 = self.pressure[i - 1, j, k]
                p_x2 = self.pressure[i + 1, j, k]
                p_y1 = self.pressure[i, j - 1, k]
                p_y2 = self.pressure[i, j + 1, k]
                p_z1 = self.pressure[i, j, k - 1]
                p_z2 = self.pressure[i, j, k + 1]
                n = 6
                if self.solver.is_colliding(i - 1, j, k):
                    p_x1 = 0.0
                    n -= 1
                if self.solver.is_colliding(i + 1, j, k):
                    p_x2 = 0.0
                    n -= 1
                if self.solver.is_colliding(i, j - 1, k):
                    p_y1 = 0.0
                    n -= 1
                if self.solver.is_colliding(i, j + 1, k):
                    p_y2 = 0.0
                    n -= 1
                if self.solver.is_colliding(i, j, k - 1):
                    p_z1 = 0.0
                    n -= 1
                if self.solver.is_colliding(i, j, k + 1):
                    p_z2 = 0.0
                    n -= 1

                s = p_x1 + p_x2 + p_y1 + p_y2 + p_z1 + p_z2
                dt, dx = self.solver.dt[None], self.solver.dx
                w, p = self.jacobi_weight, self.pressure[i, j, k]
                pressure = (1 - w) * p + w * (s - div * self.rho / dt * (dx**2)) / n

            self.pressure_new[i, j, k] = pressure

    @ti.kernel
    def apply_pressure(self):
        scale = self.solver.dt[None] / (self.rho * self.solver.dx)
        for i, j, k in ti.ndrange(self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]):
            if self.solver.is_interior(i - 1, j, k) or self.solver.is_interior(i, j, k):
                if self.solver.is_colliding(i - 1, j, k) or self.solver.is_colliding(i, j, k):
                    self.solver.velocity_x[i, j, k] = 0
                else:
                    pressure_gradient = self.pressure[i, j, k] - self.pressure[i - 1, j, k]
                    self.solver.velocity_x[i, j, k] -= scale * pressure_gradient

            if self.solver.is_interior(i, j - 1, k) or self.solver.is_interior(i, j, k):
                if self.solver.is_colliding(i, j - 1, k) or self.solver.is_colliding(i, j, k):
                    self.solver.velocity_y[i, j, k] = 0
                else:
                    pressure_gradient = self.pressure[i, j, k] - self.pressure[i, j - 1, k]
                    self.solver.velocity_y[i, j, k] -= scale * pressure_gradient

            if self.solver.is_interior(i, j, k - 1) or self.solver.is_interior(i, j, k):
                if self.solver.is_colliding(i, j, k - 1) or self.solver.is_colliding(i, j, k):
                    self.solver.velocity_z[i, j, k] = 0
                else:
                    pressure_gradient = self.pressure[i, j, k] - self.pressure[i, j, k - 1]
                    self.solver.velocity_z[i, j, k] -= scale * pressure_gradient

    def solve(self):
        self.compute_divergence()
        for _ in range(self.jacobi_iter):
            self.jacobi_iteration()
            self.pressure.copy_from(self.pressure_new)
        self.apply_pressure()
