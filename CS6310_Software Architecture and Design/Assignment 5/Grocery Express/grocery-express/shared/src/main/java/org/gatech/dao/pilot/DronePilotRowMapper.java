package org.gatech.dao.pilot;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.DronePilot;

import java.sql.ResultSet;
import java.sql.SQLException;

public class DronePilotRowMapper implements RowMapper<DronePilot> {
    /**
     * Map result set to actual DTO object
     *
     * @param rs - result set object from SQL query
     * @return Mapped entity from the result set
     * @throws SQLException - is thrown when column is not found
     */
    @Override
    public DronePilot mapRow(ResultSet rs) throws SQLException {

        return new DronePilot.DronePilotBuilder()
                .withPhoneNumber(rs.getString("phone_number"))
                .withFirstName(rs.getString("first_name"))
                .withLastName(rs.getString("last_name"))
                .withTaxId(rs.getString("tax_id"))
                .withMonthsWorkedCount(rs.getInt("months_worked_count"))
                .withSalary(rs.getInt("salary"))
                .withLicenseId(rs.getString("license_id"))
                .withAccountId(rs.getString("account_id"))
                .withSuccessfulDeliveryCount(rs.getInt("successful_delivery_count"))
                .build();
    }
}
