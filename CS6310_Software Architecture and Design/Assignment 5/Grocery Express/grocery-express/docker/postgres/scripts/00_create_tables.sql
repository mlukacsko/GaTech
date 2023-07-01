create table ge_energy_curve (
    id serial primary key,
    t_0000 numeric(2,2) not null,
    t_0100 numeric(2,2) not null,
    t_0200 numeric(2,2) not null,
    t_0300 numeric(2,2) not null,
    t_0400 numeric(2,2) not null,
    t_0500 numeric(2,2) not null,
    t_0600 numeric(2,2) not null,
    t_0700 numeric(2,2) not null,
    t_0800 numeric(2,2) not null,
    t_0900 numeric(2,2) not null,
    t_1000 numeric(2,2) not null,
    t_1100 numeric(2,2) not null,
    t_1200 numeric(2,2) not null,
    t_1300 numeric(2,2) not null,
    t_1400 numeric(2,2) not null,
    t_1500 numeric(2,2) not null,
    t_1600 numeric(2,2) not null,
    t_1700 numeric(2,2) not null,
    t_1800 numeric(2,2) not null,
    t_1900 numeric(2,2) not null,
    t_2000 numeric(2,2) not null,
    t_2100 numeric(2,2) not null,
    t_2200 numeric(2,2) not null,
    t_2300 numeric(2,2) not null
);

create table ge_settings (
    id serial primary key,
    clock time not null,
    drone_energy_consumption integer not null,
    drone_energy_curve_id integer not null references ge_energy_curve on delete restrict,
    drone_energy_capacity integer not null
);

create table ge_location (
    id serial primary key,
    x integer not null,
    y integer not null,
    unique (x, y)
);

create table ge_store (
    id serial primary key,
    name varchar(255) not null unique,
    earned_revenue integer not null default 0,
    completed_order_count integer not null default 0,
    transferred_order_count integer not null default 0,
    drone_overload_count integer not null default 0,
    location_id integer references ge_location on delete restrict
);

create table ge_item (
    id serial primary key,
    store_id integer not null references ge_store on delete restrict,
    name varchar(255) not null,
    weight integer not null default 0,
    unique (store_id, name)
);

create table ge_person (
    id serial primary key,
    phone_number varchar(255) not null,
    first_name varchar(255) not null default '',
    last_name varchar(255) not null default ''
);

create table ge_customer (
    id serial primary key,
    person_id integer not null references ge_person on delete restrict unique,
    account_id varchar(255) not null unique,
    credits integer not null default 0,
    rating integer not null default 0,
    location_id integer references ge_location on delete restrict
);

create table ge_employee (
    id serial primary key,
    person_id integer not null references ge_person on delete restrict unique,
    tax_id varchar(255) not null unique,
    months_worked_count integer not null default 0,
    salary integer not null default 0
);

create table ge_pilot (
    id serial primary key,
    employee_id integer not null references ge_employee on delete restrict unique,
    license_id varchar(255) not null unique,
    account_id varchar(255) not null unique,
    successful_delivery_count integer not null default 0
);

create table ge_drone (
    id serial primary key,
    store_id integer not null references ge_store on delete restrict,
    drone_id varchar(255) not null,
    pilot_id integer references ge_pilot on delete restrict unique,
    weight_capacity integer not null default 0,
    remaining_delivery_count integer not null default 0,
    wait_time interval hour not null default '0 hours',
    unique (store_id, drone_id)
);

create table ge_order (
    id serial primary key,
    store_id integer not null references ge_store on delete restrict,
    order_id varchar(255) not null,
    drone_id integer not null references ge_drone on delete restrict,
    customer_id integer not null references ge_customer on delete restrict,
    unique (store_id, order_id)
);

create table ge_line_item (
    id serial primary key,
    order_id integer not null references ge_order on delete restrict,
    item_id integer not null references ge_item on delete restrict,
    quantity integer not null default 0,
    unit_price integer not null default 0,
    unique (order_id, item_id)
);

create table ge_coupon (
    id serial primary key,
    customer_id integer not null references ge_customer on delete restrict,
    percentage integer not null default 0,
    expiration_date date
);