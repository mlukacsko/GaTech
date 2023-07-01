package org.gatech.service.location;

import com.google.inject.Inject;
import org.gatech.dao.location.LocationDao;
import org.gatech.dto.Location;

import java.sql.SQLException;
import java.util.List;

public class DefaultLocationService implements LocationService {
    private final LocationDao locationDao;

    @Inject
    public DefaultLocationService(LocationDao locationDao) {
        this.locationDao = locationDao;
    }

    public List<Location> getCustomerLocation(String customerAccountId) throws SQLException, ClassNotFoundException {
        return this.locationDao.getCustomerLocation(customerAccountId);
    }

    public List<Location> getStoreLocation(String storeName) throws SQLException, ClassNotFoundException {
        return this.locationDao.getStoreLocation(storeName);
    }
}
