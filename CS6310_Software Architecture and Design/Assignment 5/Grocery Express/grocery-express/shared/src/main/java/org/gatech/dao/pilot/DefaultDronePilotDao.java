package org.gatech.dao.pilot;

import com.google.inject.Inject;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.DronePilot;

import java.sql.SQLException;
import java.util.List;

public class DefaultDronePilotDao implements DronePilotDao {

    private final ConnectionManager<DronePilot> manager;
    private final DronePilotRowMapper dronePilotRowMapper;

    @Inject
    public DefaultDronePilotDao(ConnectionManager<DronePilot> manager, DronePilotRowMapper dronePilotRowMapper) {
        this.manager = manager;
        this.dronePilotRowMapper = dronePilotRowMapper;
    }


    /**
     * Gets a list of drones from the db
     *
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<DronePilot> getDronePilots() throws ClassNotFoundException, SQLException {
        return manager.getEntities("SELECT * FROM ge_pilot_person ORDER BY account_id;", dronePilotRowMapper);
    }

    /**
     * @param pilot
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void createDronePilot(DronePilot pilot) throws ClassNotFoundException, SQLException {

        // insert into the Person table
        int personId = manager.executeSqlWithReturn(
                "INSERT INTO ge_person(phone_number, first_name, last_name)" +
                        "VALUES('" + pilot.getPhoneNumber() + "','" + pilot.getFirstName() + "','" + pilot.getLastName() + "')" +
                        "RETURNING id;"
        );

        // insert into the Employee table
        int employeeId = manager.executeSqlWithReturn(
                "INSERT INTO ge_employee(person_id, tax_id, months_worked_count, salary)" +
                        "VALUES(" +
                        personId + ",'" +
                        pilot.getTaxId() + "'," +
                        pilot.getMonthsWorkedCount() + "," +
                        pilot.getSalary() + ") RETURNING id;"
        );

        // insert into the Pilot table
        manager.executeSql(
                "INSERT INTO ge_pilot(employee_id, license_id, account_id, successful_delivery_count)" +
                        "VALUES(" +
                        employeeId + ",'" +
                        pilot.getLicenseId() + "','" +
                        pilot.getAccountId() + "'," +
                        pilot.getSuccessfulDeliveryCount() + ");"
        );
    }

    /**
     * Gets a list of pilots from the db by their account identifier
     *
     * @param accountIdentifier
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<DronePilot> getDronePilotsByAccountIdentifier(String accountIdentifier) throws ClassNotFoundException, SQLException {
        return manager.getEntities(
                "SELECT * FROM ge_pilot_person WHERE account_id='" + accountIdentifier + "';",
                dronePilotRowMapper
        );
    }

    /**
     * Gets a list of pilots from the db by their license id
     *
     * @param licenseId
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<DronePilot> getDronePilotsByLicenseId(String licenseId) throws ClassNotFoundException, SQLException {
        return manager.getEntities(
                "SELECT * FROM ge_pilot_person WHERE license_id='" + licenseId + "';",
                dronePilotRowMapper
        );
    }
}
