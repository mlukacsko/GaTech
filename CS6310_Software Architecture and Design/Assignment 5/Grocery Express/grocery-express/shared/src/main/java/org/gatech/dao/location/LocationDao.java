package org.gatech.dao.location;

import org.gatech.dto.Location;

import java.sql.SQLException;
import java.util.List;

public interface LocationDao {
    List<Location> getOrderCustomerLocation(String storeName, String orderID) throws SQLException, ClassNotFoundException;
    List<Location> getStoreLocation(String storeName) throws SQLException, ClassNotFoundException;
    List<Location> getCustomerLocation(String customerAccountId) throws SQLException, ClassNotFoundException;
}
