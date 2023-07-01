package org.gatech.service.pilot;

import com.google.inject.Inject;
import org.gatech.dao.pilot.DronePilotDao;
import org.gatech.dto.DronePilot;
import org.gatech.exception.GroceryExpressException;

import java.sql.SQLException;
import java.util.List;

public class DefaultDronePilotService implements DronePilotService {

    private final DronePilotDao dronePilotDao;

    @Inject
    public DefaultDronePilotService(DronePilotDao dronePilotDao) {
        this.dronePilotDao = dronePilotDao;
    }


    /**
     * Gets a list of pilots from the dao
     *
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<DronePilot> getDronePilots() throws ClassNotFoundException, SQLException {
        return dronePilotDao.getDronePilots();
    }

    /**
     * Calls on the dao to create a Drone pilot
     *
     * @param pilot
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void createDronePilot(DronePilot pilot) throws ClassNotFoundException, SQLException, GroceryExpressException {
        if (accountIdExists(pilot.getAccountId())) {
            throw new GroceryExpressException("ERROR:pilot_identifier_already_exists");
        }

        if (licenseIdExists(pilot.getLicenseId())) {
            throw new GroceryExpressException("ERROR:pilot_license_already_exists");
        }

        dronePilotDao.createDronePilot(pilot);
    }

    private boolean accountIdExists(String accountId) throws SQLException, ClassNotFoundException {
        List<DronePilot> dronePilots = dronePilotDao.getDronePilotsByAccountIdentifier(accountId);
        return !dronePilots.isEmpty();
    }

    private boolean licenseIdExists(String licenseId) throws SQLException, ClassNotFoundException {
        List<DronePilot> dronePilots = dronePilotDao.getDronePilotsByLicenseId(licenseId);
        return !dronePilots.isEmpty();
    }
}
