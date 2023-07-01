package org.gatech.dao.location;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.Location;

import java.sql.ResultSet;
import java.sql.SQLException;

public class LocationRowMapper implements RowMapper<Location> {
    @Override
    public Location mapRow(ResultSet rs) throws SQLException {
        return new Location.LocationBuilder()
                .withX(rs.getInt("x"))
                .withY(rs.getInt("y"))
                .build();
    }
}
