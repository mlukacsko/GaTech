package org.gatech.dbconnect;

import java.sql.ResultSet;
import java.sql.SQLException;

public interface RowMapper<T extends DatabaseEntity> {

    /**
     * Map result set to actual DTO object
     * @param rs - result set object from SQL query
     * @return Mapped entity from the result set
     * @throws SQLException - is thrown when column is not found
     */
    T mapRow(ResultSet rs) throws SQLException;
}
