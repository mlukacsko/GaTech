create view ge_customer_person as
    select phone_number, first_name, last_name, account_id, credits, rating
    from ge_customer
    inner join ge_person on ge_customer.person_id = ge_person.id;

create view ge_pilot_person as
    select ge_pilot.id as pilot_id, phone_number, first_name, last_name, tax_id, months_worked_count, salary, license_id, account_id, successful_delivery_count
    from ge_pilot
    inner join ge_employee on ge_pilot.employee_id = ge_employee.id
    inner join ge_person on ge_employee.person_id = ge_person.id;
