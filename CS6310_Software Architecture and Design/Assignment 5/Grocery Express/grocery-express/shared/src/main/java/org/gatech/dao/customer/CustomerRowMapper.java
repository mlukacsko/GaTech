package org.gatech.dao.customer;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.Customer;

import java.sql.ResultSet;
import java.sql.SQLException;

public class CustomerRowMapper implements RowMapper<Customer> {
    /**
     * Map result set to actual DTO object
     *
     * @param rs - result set object from SQL query
     * @return Mapped entity from the result set
     * @throws SQLException - is thrown when column is not found
     */
    @Override
    public Customer mapRow(ResultSet rs) throws SQLException {
        return new Customer.CustomerBuilder()
                .withPhoneNumber(rs.getString("phone_number"))
                .withFirstName(rs.getString("first_name"))
                .withLastName(rs.getString("last_name"))
                .withAccountId(rs.getString("account_id"))
                .withCredits(rs.getInt("credits"))
                .withRating(rs.getInt("rating"))
                .build();
    }
}
