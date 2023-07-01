package org.gatech.dao.store;

import com.google.inject.Inject;
import org.gatech.dto.Store;
import org.gatech.dbconnect.ConnectionManager;

import java.sql.SQLException;
import java.util.List;

public class DefaultStoreDao implements StoreDao {

    private final ConnectionManager<Store> manager;
    private final StoreRowMapper storeRowMapper;

    @Inject
    public DefaultStoreDao(ConnectionManager<Store> manager, StoreRowMapper storeRowMapper) {
        this.manager = manager;
        this.storeRowMapper = storeRowMapper;
    }

    @Override
    public List<Store> getStores() throws SQLException, ClassNotFoundException {
        return manager.getEntities("SELECT * FROM ge_store ORDER BY name ASC;", storeRowMapper);
    }

    /**
     * Creates a store in the database
     *
     * @param store - the store to be created
     * @throws SQLException           - is thrown if SQL is not valid
     * @throws ClassNotFoundException - is thrown if driver is not found
     */
    @Override
    public void createStore(Store store) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "INSERT INTO GE_STORE(NAME, EARNED_REVENUE, COMPLETED_ORDER_COUNT, TRANSFERRED_ORDER_COUNT, DRONE_OVERLOAD_COUNT)" +
                        "VALUES('" + store.getName() + "', " + store.getEarnedRevenue() + ", " + store.getCompletedOrderCount() + "," +
                         store.getTransferredOrderCount() + "," + store.getDroneOverloadCount() + ");"

        );
    }

    /**
     * @param storeName
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public List<Store> getStoresByName(String storeName) throws SQLException, ClassNotFoundException {
        return manager.getEntities("SELECT * FROM ge_store WHERE name='" + storeName + "';", storeRowMapper);
    }
}
