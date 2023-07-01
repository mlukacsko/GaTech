package org.gatech.service.location;

import org.gatech.dto.Location;

import java.sql.SQLException;
import java.util.List;

public interface LocationService {
    public List<Location> getCustomerLocation(String customerAccountId) throws SQLException, ClassNotFoundException;
    public List<Location> getStoreLocation(String storeName) throws SQLException, ClassNotFoundException;
}
