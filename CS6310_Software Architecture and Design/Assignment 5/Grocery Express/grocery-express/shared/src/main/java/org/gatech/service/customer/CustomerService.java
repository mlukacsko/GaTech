package org.gatech.service.customer;

import org.gatech.dto.Customer;
import org.gatech.exception.GroceryExpressException;

import java.sql.SQLException;
import java.util.List;

public interface CustomerService {

    /**
     * Gets a list of customers from dao
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<Customer> getCustomers() throws ClassNotFoundException, SQLException;

    /**
     * Calls upon dao to create a customer
     * @param customer
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    void createCustomer(Customer customer) throws ClassNotFoundException, SQLException, GroceryExpressException;

    void distributeCoupons(int numberOfCoupons) throws ClassNotFoundException, SQLException;
}
