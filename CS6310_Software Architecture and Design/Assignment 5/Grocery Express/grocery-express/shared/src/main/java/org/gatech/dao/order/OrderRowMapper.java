package org.gatech.dao.order;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.Order;

import java.sql.ResultSet;
import java.sql.SQLException;

public class OrderRowMapper implements RowMapper<Order> {
    @Override
    public Order mapRow(ResultSet rs) throws SQLException {
        return new Order.OrderBuilder()
                .withID(rs.getString("id"))
                .withStoreID(rs.getInt("store_id"))
                .withOrderID(rs.getString("order_id"))
                .withDroneID(rs.getInt("drone_id"))
                .withCustomerID(rs.getString("customer_id"))
                .build();
    }
}
