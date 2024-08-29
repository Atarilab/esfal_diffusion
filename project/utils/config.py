class Go2Config:
    # Name of the robot in robot descriptions repo
    name = "go2"
    # Local mesh dir
    mesh_dir = "assets"
    # Rotor ineretia (optional)
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    # Gear ratio (optional)
    gear_ratio = 6.33