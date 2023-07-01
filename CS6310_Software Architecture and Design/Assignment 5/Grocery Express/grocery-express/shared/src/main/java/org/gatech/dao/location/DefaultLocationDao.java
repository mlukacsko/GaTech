package org.gatech.dao.location;

import com.google.inject.Inject;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.Location;

import java.sql.SQLException;
import java.util.List;

public class DefaultLocationDao implements LocationDao {
    private final ConnectionManager<Location> manager;
    private final LocationRowMapper locationRowMapper;

    @Inject
    public DefaultLocationDao(ConnectionManager<Location> manager, LocationRowMapper locationRowMapper) {
        this.manager = manager;
        this.locationRowMapper = locationRowMapper;
    }

    @Override
    public List<Location> getOrderCustomerLocation(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        return this.manager.getEntities(
                "SELECT x, y FROM ge_order " +
                    "INNER JOIN ge_store ON ge_order.store_id = ge_store.id " +
                    "INNER JOIN ge_customer ON ge_order.customer_id = ge_customer.id " +
                    "INNER JOIN ge_location ON ge_customer.location_id = ge_location.id " +
                    "WHERE ge_store.name = '" + storeName + "' " +
                    "AND ge_order.order_id = '" + orderID + "';",
                locationRowMapper);
    }

    public List<Location> getStoreLocation(String storeName) throws SQLException, ClassNotFoundException {
        return this.manager.getEntities(
                "SELECT x, y FROM ge_store " +
                        "INNER JOIN ge_location ON ge_store.location_id = ge_location.id " +
                        "WHERE ge_store.name = '" + storeName + "';",
                locationRowMapper);
    }

    public List<Location> getCustomerLocation(String customerAccountId) throws SQLException, ClassNotFoundException {
        return this.manager.getEntities(
            "SELECT x, y FROM ge_customer " +
                "INNER JOIN ge_location ON ge_customer.location_id = ge_location.id " +
                "WHERE ge_customer.account_id = '" + customerAccountId + "';",
                locationRowMapper
        );
    }
}
