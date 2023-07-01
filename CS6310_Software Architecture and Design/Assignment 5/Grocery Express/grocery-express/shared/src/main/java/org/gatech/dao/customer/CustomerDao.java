package org.gatech.dao.customer;

import org.gatech.dto.Customer;

import java.sql.SQLException;
import java.util.List;

public interface CustomerDao {

    /**
     * Gets a list of customer objects from the db
     * @return a list of customers
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<Customer> getCustomers() throws ClassNotFoundException, SQLException;

    /**
     * Creates a customer in the db
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    void createCustomer(Customer customer) throws ClassNotFoundException, SQLException;

    /**
     * Gets a list of customers by the account identifier
     * @param accountId
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    List<Customer> getCustomersByAccountId(String accountId) throws ClassNotFoundException, SQLException;

    /**
     *
     * @param storeName
     * @param orderIdentifier
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    boolean canCustomerAffordNewItem(String storeName, String orderIdentifier, int futureCost) throws ClassNotFoundException, SQLException;
}
