from src.solvers.collocated_solver import CollocatedSolver

import taichi as ti


@ti.data_oriented
class StaggeredSolver(CollocatedSolver):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        super().__init__(max_particles, n_grid, vol_0)

        # Properties on MAC-faces:
        self.velocity_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.velocity_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)
        self.volume_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.volume_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)
        self.mass_x = ti.field(dtype=ti.f32, shape=(self.w_grid + 1, self.w_grid), offset=self.w_offset)
        self.mass_y = ti.field(dtype=ti.f32, shape=(self.w_grid, self.w_grid + 1), offset=self.w_offset)

    @ti.func
    def left_hand_offset(self, i: ti.i32, j: ti.i32) -> ti.f32:  # pyright: ignore
        """
        Computes the initial value of the left-hand side of the pressure equation.
        This is usually zero, but might need to be offset to allow for some compression.
        """
        return 0.0

    @ti.func
    def right_hand_offset(self, i: ti.i32, j: ti.i32) -> ti.f32:  # pyright: ignore
        """
        Computes the initial value of the right-hand side of the pressure equation.
        This is usually zero, but might need to be offset to allow for some compression.
        """
        return 0.0
