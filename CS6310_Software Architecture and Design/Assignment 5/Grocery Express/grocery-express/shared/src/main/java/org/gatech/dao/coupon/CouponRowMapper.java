package org.gatech.dao.coupon;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.Coupon;

import java.sql.ResultSet;
import java.sql.SQLException;

public class CouponRowMapper implements RowMapper<Coupon> {
    /**
     * Map result set to actual DTO object
     *
     * @param rs - result set object from SQL query
     * @return Mapped entity from the result set
     * @throws SQLException - is thrown when column is not found
     */
    @Override
    public Coupon mapRow(ResultSet rs) throws SQLException {
        return new Coupon.CouponBuilder()
                .withCustomerId(rs.getInt("customer_id"))
                .withPercentage(rs.getInt("percentage"))
                .withExpirationDate(rs.getDate("expiration_date"))
                .build();
    }
}
