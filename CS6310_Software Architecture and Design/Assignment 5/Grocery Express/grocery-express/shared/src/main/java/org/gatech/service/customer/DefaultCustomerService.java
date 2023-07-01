package org.gatech.service.customer;

import com.google.inject.Inject;
import org.gatech.dao.coupon.CouponDao;
import org.gatech.dao.customer.CustomerDao;
import org.gatech.dto.Customer;
import org.gatech.exception.GroceryExpressException;

import java.sql.SQLException;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class DefaultCustomerService implements CustomerService {

    private final CustomerDao customerDao;
    private final CouponDao couponDao;

    @Inject
    public DefaultCustomerService(CustomerDao customerDao, CouponDao couponDao) {
        this.customerDao = customerDao;
        this.couponDao = couponDao;
    }

    /**
     * Gets a list of customers from dao
     *
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<Customer> getCustomers() throws ClassNotFoundException, SQLException {
        return customerDao.getCustomers();
    }

    /**
     * Calls upon dao to create a customer
     *
     * @param customer
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void createCustomer(Customer customer) throws ClassNotFoundException, SQLException, GroceryExpressException {
        if (customerExists(customer.getAccountId())) {
            throw new GroceryExpressException("ERROR:customer_identifier_already_exists");
        }
        customerDao.createCustomer(customer);
    }

    /**
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void distributeCoupons(int numberOfCoupons) throws ClassNotFoundException, SQLException {
        List<Customer> customers = customerDao.getCustomers(); // get all the customers in the db
        Collections.shuffle(customers); // shuffle the list
        List<Customer> luckyCustomers = customers.stream().limit(15).collect(Collectors.toList());

        couponDao.distributeCoupons(luckyCustomers, numberOfCoupons);

    }

    private boolean customerExists(String accountId) throws SQLException, ClassNotFoundException {
        List<Customer> customers = customerDao.getCustomersByAccountId(accountId);
        return !customers.isEmpty();
    }
}
