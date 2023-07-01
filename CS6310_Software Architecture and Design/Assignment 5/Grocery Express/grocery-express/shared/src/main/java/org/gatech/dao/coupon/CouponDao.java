package org.gatech.dao.coupon;

import org.gatech.dto.Coupon;
import org.gatech.dto.Customer;

import java.sql.SQLException;
import java.util.List;

public interface CouponDao {

    void distributeCoupons(List<Customer> luckyCustomers, int numberOfCoupons) throws ClassNotFoundException, SQLException;

    List<Coupon> getApplicableCoupons(String storeName, String orderId) throws ClassNotFoundException, SQLException;
}
