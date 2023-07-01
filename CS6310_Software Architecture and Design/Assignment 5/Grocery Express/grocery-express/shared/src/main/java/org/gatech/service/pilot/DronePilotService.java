package org.gatech.service.pilot;

import org.gatech.dto.DronePilot;
import org.gatech.exception.GroceryExpressException;

import java.sql.SQLException;
import java.util.List;

public interface DronePilotService {

    /**
     * Gets a list of pilots from the dao
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<DronePilot> getDronePilots() throws ClassNotFoundException, SQLException;

    /**
     * Calls on the dao to create a Drone pilot
     * @param pilot
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    void createDronePilot(DronePilot pilot) throws ClassNotFoundException, SQLException, GroceryExpressException;
}
