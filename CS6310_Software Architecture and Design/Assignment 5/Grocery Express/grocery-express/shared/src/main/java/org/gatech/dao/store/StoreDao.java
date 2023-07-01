package org.gatech.dao.store;

import org.gatech.dto.Store;

import java.sql.SQLException;
import java.util.List;

public interface StoreDao {

    /**
     * Retrieves a list of stores from the database
     * @return a list of stores
     * @throws SQLException - is thrown if SQL is invalid
     * @throws ClassNotFoundException - is thrown if proper db class driver is not found
     */
    List<Store> getStores() throws SQLException, ClassNotFoundException;

    /**
     * Creates a store in the database
     * @param store - the store to be created
     * @throws SQLException - is thrown if SQL is not valid
     * @throws ClassNotFoundException - is thrown if driver is not found
     */
    void createStore(Store store) throws SQLException, ClassNotFoundException;

    List<Store> getStoresByName(String storeName) throws SQLException, ClassNotFoundException;
}
