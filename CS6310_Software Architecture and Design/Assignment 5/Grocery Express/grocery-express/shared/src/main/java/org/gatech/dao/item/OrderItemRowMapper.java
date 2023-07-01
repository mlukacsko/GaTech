package org.gatech.dao.item;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.OrderItem;

import java.sql.ResultSet;
import java.sql.SQLException;

public class OrderItemRowMapper implements RowMapper<OrderItem> {
    /**
     * Map result set to actual DTO object
     *
     * @param rs - result set object from SQL query
     * @return Mapped entity from the result set
     * @throws SQLException - is thrown when column is not found
     */
    @Override
    public OrderItem mapRow(ResultSet rs) throws SQLException {
        return new OrderItem.OrderItemBuilder()
                .withItemName(rs.getString("item_name"))
                .withTotalQuantity(rs.getInt("total_quantity"))
                .withTotalCost(rs.getInt("total_cost"))
                .withTotalWeight(rs.getInt("total_weight"))
                .build();
    }
}
