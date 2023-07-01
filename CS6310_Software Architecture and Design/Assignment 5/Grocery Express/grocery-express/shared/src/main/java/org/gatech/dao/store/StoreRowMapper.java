package org.gatech.dao.store;

import org.gatech.dto.Store;
import org.gatech.dbconnect.RowMapper;

import java.sql.ResultSet;
import java.sql.SQLException;

public class StoreRowMapper implements RowMapper<Store> {
    @Override
    public Store mapRow(ResultSet rs) throws SQLException {
        return new Store.StoreBuilder()
                .withName(rs.getString("name"))
                .withEarnedRevenue(rs.getInt("earned_revenue"))
                .withCompletedOrderCount(rs.getInt("completed_order_count"))
                .withTransferredOrderCount(rs.getInt("transferred_order_count"))
                .withDroneOverloadCount(rs.getInt("drone_overload_count"))
                .build();
    }
}
