from src.configurations import Circle, Configuration
from src.constants import Ice

ice_presets = [
    Configuration(
        name="Spherefall, Ice",
        information="Ice",
        gravity=-9.81,
        dt=5e-4,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.5, 0.4, 0.5),
                velocity=(0, -3, 0),
                temperature=-100.0,
                radius=0.1,
            ),
        ],
    ),
]
