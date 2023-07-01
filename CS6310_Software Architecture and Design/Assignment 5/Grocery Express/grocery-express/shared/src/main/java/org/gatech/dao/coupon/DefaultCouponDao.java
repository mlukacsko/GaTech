package org.gatech.dao.coupon;

import com.google.inject.Inject;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.Coupon;
import org.gatech.dto.Customer;

import java.sql.SQLException;
import java.util.List;

public class DefaultCouponDao implements CouponDao {

    private final ConnectionManager<Coupon> connectionManager;
    private final CouponRowMapper rowMapper;

    @Inject
    public DefaultCouponDao(ConnectionManager<Coupon> connectionManager, CouponRowMapper rowMapper) {
        this.connectionManager = connectionManager;
        this.rowMapper = rowMapper;
    }

    /**
     * @param luckyCustomers
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void distributeCoupons(List<Customer> luckyCustomers, int numberOfCoupons) throws ClassNotFoundException, SQLException {
        int couponCount = numberOfCoupons;
        for (Customer c : luckyCustomers) {
            if (couponCount <= 0) {
                break;
            }
            int customerId = connectionManager.executeSqlWithReturn(
                    "SELECT id FROM ge_customer WHERE account_id='" + c.getAccountId() + "';"
            );

            if (c.getRating() > 3) {  // if the customer is good, give them an extra coupon
                connectionManager.executeSql(
                        "INSERT INTO ge_coupon(customer_id, percentage, expiration_date)" +
                                "VALUES(" + customerId + "," + "15, NOW() + INTERVAL '30 days');"
                );
            }

            connectionManager.executeSql(
                    "INSERT INTO ge_coupon(customer_id, percentage, expiration_date)" +
                            "VALUES(" + customerId + "," + "15, NOW() + INTERVAL '30 days');"
            );
            --couponCount;
        }
    }

    /**
     * @param storeName
     * @param orderId
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<Coupon> getApplicableCoupons(String storeName, String orderId) throws ClassNotFoundException, SQLException {
        return connectionManager.getEntities(
                "SELECT coupon.* " +
                        "FROM ge_coupon coupon " +
                        "INNER JOIN ge_order o ON o.customer_id=coupon.customer_id " +
                        "INNER JOIN ge_store s ON s.id=o.store_id " +
                        "WHERE s.name='" + storeName + "' " +
                        "  AND o.order_id='" + orderId + "';",
                rowMapper
        );
    }
}
