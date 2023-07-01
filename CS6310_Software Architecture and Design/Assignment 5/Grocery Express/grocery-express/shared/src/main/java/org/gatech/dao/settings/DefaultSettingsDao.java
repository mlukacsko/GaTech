package org.gatech.dao.settings;

import com.google.inject.Inject;
import org.gatech.dao.drone.DroneDao;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.Drone;
import org.gatech.dto.Settings;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;

public class DefaultSettingsDao implements SettingsDao {
    private final ConnectionManager<Settings> manager;
    private final SettingsRowMapper settingsRowMapper;
    private final DroneDao droneDao;

    @Inject
    public DefaultSettingsDao(ConnectionManager<Settings> manager, SettingsRowMapper settingsRowMapper, DroneDao droneDao) {
        this.manager = manager;
        this.settingsRowMapper = settingsRowMapper;
        this.droneDao = droneDao;
    }

    public HashMap<Integer, Float> getSolarEnergyMap() throws SQLException, ClassNotFoundException {
        Settings settings = getSettings().get(0);
        HashMap<Integer, Float> solarEnergyMap = new HashMap<>();
        solarEnergyMap.put(0, Math.abs(settings.droneEnergyRestorationT0000));
        solarEnergyMap.put(1, Math.abs(settings.droneEnergyRestorationT0100));
        solarEnergyMap.put(2, Math.abs(settings.droneEnergyRestorationT0200));
        solarEnergyMap.put(3, Math.abs(settings.droneEnergyRestorationT0300));
        solarEnergyMap.put(4, Math.abs(settings.droneEnergyRestorationT0400));
        solarEnergyMap.put(5, Math.abs(settings.droneEnergyRestorationT0500));
        solarEnergyMap.put(6, Math.abs(settings.droneEnergyRestorationT0600));
        solarEnergyMap.put(7, Math.abs(settings.droneEnergyRestorationT0700));
        solarEnergyMap.put(8, Math.abs(settings.droneEnergyRestorationT0800));
        solarEnergyMap.put(9, Math.abs(settings.droneEnergyRestorationT0900));
        solarEnergyMap.put(10,Math.abs(settings.droneEnergyRestorationT1000));
        solarEnergyMap.put(11,Math.abs(settings.droneEnergyRestorationT1100));
        solarEnergyMap.put(12,Math.abs(settings.droneEnergyRestorationT1200));
        solarEnergyMap.put(13,Math.abs(settings.droneEnergyRestorationT1300));
        solarEnergyMap.put(14,Math.abs(settings.droneEnergyRestorationT1400));
        solarEnergyMap.put(15,Math.abs(settings.droneEnergyRestorationT1500));
        solarEnergyMap.put(16,Math.abs(settings.droneEnergyRestorationT1600));
        solarEnergyMap.put(17,Math.abs(settings.droneEnergyRestorationT1700));
        solarEnergyMap.put(18,Math.abs(settings.droneEnergyRestorationT1800));
        solarEnergyMap.put(19,Math.abs(settings.droneEnergyRestorationT1900));
        solarEnergyMap.put(20,Math.abs(settings.droneEnergyRestorationT2000));
        solarEnergyMap.put(21,Math.abs(settings.droneEnergyRestorationT2100));
        solarEnergyMap.put(22,Math.abs(settings.droneEnergyRestorationT2200));
        solarEnergyMap.put(23,Math.abs(settings.droneEnergyRestorationT2300));
        return solarEnergyMap;
    }

    @Override
    public List<Settings> getSettings() throws SQLException, ClassNotFoundException {
        return manager.getEntities(
            "SELECT * FROM ge_settings " +
                "INNER JOIN ge_energy_curve " +
                "ON ge_settings.drone_energy_curve_id = ge_energy_curve.id;",
                settingsRowMapper);
    }

    @Override
    public void advanceTime(int newHour) throws SQLException, ClassNotFoundException {
        List<Drone> drones = droneDao.getAllAvailableDrones();
        List<Settings> settings = getSettings();
        String clock = settings.get(0).getClock();

        int oldHour = Integer.parseInt(clock.split(":")[0]);

        int hoursToAdvance = newHour >= oldHour ? newHour - oldHour : 24 - Math.abs(newHour - oldHour);

        int droneEnergyConsumption = settings.get(0).getDroneEnergyConsumption();
        int droneEnergyCapacity = settings.get(0).getDroneEnergyCapacity();

        // Increment energy of all available drones in the system
        // Drones that are unavailable due to servicing deliveries
        // have already had their energy levels adjusted according
        // to hours they will be busy
        double energyGain = 0.0;

        for (int i = oldHour; i < oldHour + hoursToAdvance; i++) {
            int hour = i <= 23 ? i : i % 24;
            energyGain += droneEnergyConsumption * getSolarEnergyMap().get(hour);
        }

        int floorEnergyGain = (int) Math.floor(energyGain);

        manager.executeSql(
                "UPDATE ge_drone " +
                "SET remaining_delivery_count = LEAST(remaining_delivery_count + " + floorEnergyGain + ", " + droneEnergyCapacity + ") " +
                "WHERE wait_time = '00:00:00';"
        );

        // Update wait times for drones that are not available
        // An oversight here if a drone becomes available during
        // the wait interval it will not get any additional charge
        // for the extra hours it is available.
        manager.executeSql(
                "UPDATE ge_drone " +
                "SET wait_time = wait_time - LEAST(wait_time, '" + hoursToAdvance + "hours') " +
                "WHERE wait_time != '00:00:00';"
        );

        // Update grocery express clock to the requested time
        manager.executeSql(
                "UPDATE ge_settings SET clock = '" + newHour + ":00" + "';"
        );
    }

    @Override
    public void setDroneEnergyCost(String energyCost) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "UPDATE ge_settings SET drone_energy_consumption = " + energyCost + ";"
        );
    }

    @Override
    public void setDroneEnergyCapacity(String energyCapacity) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "UPDATE ge_settings SET drone_energy_capacity = " + energyCapacity + ";"
        );
    }

    @Override
    public void setDroneEnergyRestoration(String time, String energyRestoration) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "UPDATE ge_energy_curve SET t_" + time + " = " + energyRestoration + ";"
        );
    }

    private int upsertLocation(String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException {
        return manager.executeSqlWithReturn(
                "WITH result AS (" +
                        "INSERT INTO ge_location (x, y) " +
                        "VALUES (" + xCoordinate + "," + yCoordinate + ") " +
                        "ON CONFLICT DO NOTHING " +
                        "RETURNING id" +
                ") " +
                "SELECT id FROM result " +
                "UNION " +
                "SELECT id FROM ge_location WHERE x = " + xCoordinate + " AND y = " + yCoordinate + ";"
        );
    }


    @Override
    public void setStoreLocation(String storeName, String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException {
        int locationId = upsertLocation(xCoordinate, yCoordinate);

        manager.executeSql(
                "UPDATE ge_store SET location_id = " + locationId + " " +
                "WHERE name = '" + storeName + "';"
        );
    }

    @Override
    public void setCustomerLocation(String customerAccountId, String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException {
        int locationId = upsertLocation(xCoordinate, yCoordinate);

        manager.executeSql(
                "UPDATE ge_customer SET location_id = " + locationId + " " +
                "WHERE account_id = '" + customerAccountId + "';"
        );
    }
}
