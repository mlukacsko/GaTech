package org.gatech.service.settings;

import com.google.inject.Inject;
import org.gatech.dao.settings.SettingsDao;
import org.gatech.dto.Settings;

import java.sql.SQLException;
import java.util.List;

public class DefaultSettingsService implements SettingsService {
    private final SettingsDao settingsDao;

    @Inject
    public DefaultSettingsService(SettingsDao settingsDao) {
        this.settingsDao = settingsDao;
    }

    @Override
    public List<Settings> getSettings() throws SQLException, ClassNotFoundException {
        return this.settingsDao.getSettings();
    }

    @Override
    public void advanceTime(String hour) throws SQLException, ClassNotFoundException {
        this.settingsDao.advanceTime(Integer.parseInt(hour));
    }

    @Override
    public void setDroneEnergyCost(String energyCost) throws SQLException, ClassNotFoundException {
        this.settingsDao.setDroneEnergyCost(energyCost);
    }

    @Override
    public void setDroneEnergyCapacity(String energyCapacity) throws SQLException, ClassNotFoundException {
        this.settingsDao.setDroneEnergyCapacity(energyCapacity);
    }

    @Override
    public void setDroneEnergyRestoration(String time, String energyCapacity) throws SQLException, ClassNotFoundException {
        this.settingsDao.setDroneEnergyRestoration(time, energyCapacity);
    }

    @Override
    public void setStoreLocation(String storeName, String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException {
        this.settingsDao.setStoreLocation(storeName, xCoordinate, yCoordinate);
    }

    @Override
    public void setCustomerLocation(String customerAccountId, String xCoordinate, String yCoordinate) throws SQLException, ClassNotFoundException {
        this.settingsDao.setCustomerLocation(customerAccountId, xCoordinate, yCoordinate);
    }
}
