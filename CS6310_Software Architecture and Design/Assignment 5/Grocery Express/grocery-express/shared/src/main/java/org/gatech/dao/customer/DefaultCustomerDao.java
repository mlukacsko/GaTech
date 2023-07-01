package org.gatech.dao.customer;

import com.google.inject.Inject;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.Customer;

import java.sql.SQLException;
import java.util.List;

public class DefaultCustomerDao implements CustomerDao {

    private final ConnectionManager<Customer> manager;
    private final CustomerRowMapper customerRowMapper;

    @Inject
    public DefaultCustomerDao(ConnectionManager<Customer> manager, CustomerRowMapper customerRowMapper) {
        this.manager = manager;
        this.customerRowMapper = customerRowMapper;
    }

    /**
     * Gets a list of customer objects from the db
     *
     * @return a list of customers
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<Customer> getCustomers() throws ClassNotFoundException, SQLException {
        return manager.getEntities("SELECT * FROM ge_customer_person ORDER BY account_id ASC;", customerRowMapper);
    }

    /**
     * Creates a customer in the db
     *
     * @param customer
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void createCustomer(Customer customer) throws ClassNotFoundException, SQLException {

        // insert info into the person table
        int personId = manager.executeSqlWithReturn(
                "INSERT INTO ge_person(phone_number, first_name, last_name)" +
                        "VALUES('" + customer.getPhoneNumber() + "','" + customer.getFirstName() + "','" + customer.getLastName() + "')" +
                        "RETURNING id;"
        );

        // insert info into the customer table
        manager.executeSql(
                "INSERT INTO ge_customer(person_id, account_id, credits, rating)" +
                        "VALUES(" +
                        personId + ",'" +
                        customer.getAccountId() + "'," +
                        customer.getCredits() + "," +
                        customer.getRating() + ");"
        );

    }

    /**
     * Gets a list of customers by the account identifier
     *
     * @param accountId
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public List<Customer> getCustomersByAccountId(String accountId) throws ClassNotFoundException, SQLException {
        return manager.getEntities(
                "SELECT * FROM ge_customer_person WHERE account_id='" + accountId + "';",
                customerRowMapper
        );
    }

    /**
     * @param storeName
     * @param orderIdentifier
     * @return
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public boolean canCustomerAffordNewItem(String storeName, String orderIdentifier, int futureCost) throws ClassNotFoundException, SQLException {
        int customerId = manager.executeSqlWithReturn(
                "SELECT o.customer_id AS id " +
                        "FROM ge_order o " +
                        "INNER JOIN ge_store s ON o.store_id=s.id " +
                        "WHERE s.name='" + storeName + "' " +
                        "  AND o.order_id='" + orderIdentifier + "';"
        );

        int sum = manager.executeSqlWithReturn(
                "SELECT SUM(li.quantity * li.unit_price) AS id " +
                        "FROM ge_line_item li " +
                        "INNER JOIN ge_order o ON o.id=li.order_id " +
                        "WHERE o.customer_id=" + customerId + ";"
        );

        int totalPendingCost = sum + futureCost;

        int credits = manager.executeSqlWithReturn(
                "SELECT credits AS id FROM ge_customer WHERE id=" + customerId + ";"
        );

        return credits >= totalPendingCost;
    }
}
