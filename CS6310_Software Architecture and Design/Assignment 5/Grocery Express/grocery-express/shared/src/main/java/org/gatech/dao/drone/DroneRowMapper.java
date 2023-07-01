package org.gatech.dao.drone;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.Drone;
import org.gatech.dto.Item;

import java.sql.ResultSet;
import java.sql.SQLException;

public class DroneRowMapper implements RowMapper<Drone> {
    @Override
    public Drone mapRow(ResultSet rs) throws SQLException {
        return new Drone.DroneBuilder()
                .withDroneID(rs.getString("drone_id"))
                .withWeightCapacity(rs.getInt("weight_capacity"))
                .withNumDeliveriesBeforeMaintenance(rs.getInt("remaining_delivery_count"))
                .withNumberOfOrders(rs.getInt("assigned_orders_count"))
                .withPilotFirstName(rs.getString("first_name"))
                .withPilotLastName(rs.getString("last_name"))
                .withTotalOrderWeight(rs.getInt("assigned_orders_weight"))
                .withWaitTime(rs.getString("wait_time"))
                .build();
    }
}
