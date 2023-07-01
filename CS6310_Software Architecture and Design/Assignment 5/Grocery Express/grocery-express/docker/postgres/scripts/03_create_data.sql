insert into ge_energy_curve (
    id,
    t_0000,t_0100,t_0200,t_0300,t_0400,t_0500,t_0600,t_0700,t_0800,t_0900,t_1000,t_1100,
    t_1200,t_1300,t_1400,t_1500,t_1600,t_1700,t_1800,t_1900,t_2000,t_2100,t_2200,t_2300
) VALUES (
    1,
    0.00,0.00,0.00,0.00,0.00,0.01,0.05,0.07,0.09,0.10,0.15,0.20,
    0.30,0.30,0.30,0.30,0.25,0.20,0.20,0.15,0.05,0.01,0.00,0.00
);

insert into ge_settings (
    clock,
    drone_energy_consumption,
    drone_energy_curve_id,
    drone_energy_capacity
) values (
    '12:00',
    1,
    1,
    40
);