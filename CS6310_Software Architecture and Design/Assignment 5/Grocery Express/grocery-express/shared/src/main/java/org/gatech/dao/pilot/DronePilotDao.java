package org.gatech.dao.pilot;

import org.gatech.dto.DronePilot;

import java.sql.SQLException;
import java.util.List;

public interface DronePilotDao {

    /**
     * Gets a list of drones from the db
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<DronePilot> getDronePilots() throws ClassNotFoundException, SQLException;

    void createDronePilot(DronePilot pilot) throws ClassNotFoundException, SQLException;

    /**
     * Gets a list of pilots from the db by their account identifier
     *
     * @param accountIdentifier
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<DronePilot> getDronePilotsByAccountIdentifier(String accountIdentifier) throws ClassNotFoundException, SQLException;

    /**
     * Gets a list of pilots from the db by their license id
     *
     * @param licenseId
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<DronePilot> getDronePilotsByLicenseId(String licenseId) throws ClassNotFoundException, SQLException;
}
