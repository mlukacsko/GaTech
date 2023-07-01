package org.gatech.service.settings;

import org.gatech.dto.Settings;

import java.sql.SQLException;
import java.util.List;

public interface SettingsService {
    List<Settings> getSettings() throws SQLException, ClassNotFoundException;
    void advanceTime(String hour) throws SQLException, ClassNotFoundException;
    void setDroneEnergyCost(String energyCost) throws SQLException, ClassNotFoundException;
    void setDroneEnergyCapacity(String energyCapacity) throws SQLException, ClassNotFoundException;
    void setDroneEnergyRestoration(String time, String energyRestoration) throws SQLException, ClassNotFoundException;
    void setStoreLocation(String storeName, String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException;
    void setCustomerLocation(String customerAccountId, String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException;
}
