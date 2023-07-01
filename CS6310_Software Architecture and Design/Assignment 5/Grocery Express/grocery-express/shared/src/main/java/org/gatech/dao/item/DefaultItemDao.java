package org.gatech.dao.item;

import com.google.inject.Inject;
import org.gatech.dto.Item;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.OrderItem;
import org.gatech.dto.Store;

import java.sql.SQLException;
import java.util.List;

public class DefaultItemDao implements ItemDao {

    private final ConnectionManager<Item> manager;
    private final ConnectionManager<OrderItem> orderItemConnectionManager;
    private final ItemRowMapper itemRowMapper;
    private final OrderItemRowMapper orderItemRowMapper;

    @Inject
    public DefaultItemDao(ConnectionManager<Item> manager, ConnectionManager<OrderItem> orderItemConnectionManager, ItemRowMapper itemRowMapper, OrderItemRowMapper orderItemRowMapper) {
        this.manager = manager;
        this.orderItemConnectionManager = orderItemConnectionManager;
        this.itemRowMapper = itemRowMapper;
        this.orderItemRowMapper = orderItemRowMapper;
    }

    @Override
    public List<Item> getItems(String storeName) throws SQLException, ClassNotFoundException {
        return manager.getEntities("SELECT i.name as item_name, s.name as store_name, i.weight as item_weight FROM ge_item i, ge_store s " +
                "WHERE s.name = '" + storeName + "' " +
                "AND i.store_id = s.id " +
                "ORDER BY item_name ASC;", itemRowMapper);
    }


    public void sellItem(Item item) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "INSERT INTO ge_item(store_id,name,weight)" +
                        "VALUES((SELECT id FROM ge_store WHERE name = '" + item.getName() + "')," + "'" +
                        item.getItemName() + "'," + item.getItemWeight() + ");"
        );
    }

    public List<Item> getItemByName(String itemName, String storeName) throws SQLException, ClassNotFoundException {
        return manager.getEntities("SELECT i.name as item_name, i.weight as item_weight, s.name as store_name FROM ge_item i, ge_store s " +
                "WHERE s.name = '" + storeName + "' " +
                "AND i.name = '" + itemName + "' "+
                "AND i.store_id = s.id;", itemRowMapper);
    }

    /**
     * @param storeName
     * @param orderIdentifier
     * @param itemName
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public List<Item> getItemsByOrderIdentifier(String storeName, String orderIdentifier, String itemName) throws SQLException, ClassNotFoundException {
        return manager.getEntities(
            "SELECT i.name as item_name, i.weight as item_weight, s.name as store_name " +
                    "FROM ge_line_item li " +
                    "INNER JOIN ge_item i ON li.item_id=i.id " +
                    "INNER JOIN ge_store s ON s.id=i.store_id " +
                    "INNER JOIN ge_order o ON li.order_id=o.id " +
                    "WHERE s.name='" + storeName + "' " +
                    "AND o.order_id='" + orderIdentifier + "' " +
                    "AND i.name = '" + itemName + "';",
                itemRowMapper
        );
    }

    /**
     * @param storeName
     * @param orderIdentifier
     * @param itemName
     * @param quantity
     * @param unitPrice
     * @throws ClassNotFoundException
     * @throws SQLException
     */
    @Override
    public void addItemToOrder(
            String storeName,
            String orderIdentifier,
            String itemName,
            int quantity,
            int unitPrice
    ) throws ClassNotFoundException, SQLException {

        int orderId = manager.executeSqlWithReturn(
                "SELECT o.id AS id " +
                        "FROM ge_order o " +
                        "INNER JOIN ge_store s ON s.id=o.store_id " +
                        "WHERE s.name='" + storeName + "' " +
                        "  AND o.order_id='" + orderIdentifier + "';"
        );

        int itemId = manager.executeSqlWithReturn(
                "SELECT i.id AS id " +
                        "FROM ge_item i " +
                        "INNER JOIN ge_store s ON s.id=i.store_id " +
                        "WHERE s.name='" + storeName + "' " +
                        "  AND i.name='" + itemName + "';"
        );

        manager.executeSql(
                "INSERT INTO ge_line_item(order_id, item_id, quantity, unit_price) " +
                        "VALUES(" + orderId + "," + itemId + "," + quantity + "," + unitPrice + ");"
        );

    }
    public List<Item> itemsExist(String storeName, String orderID) throws SQLException, ClassNotFoundException{
        return manager.getEntities("SELECT l.*, l.id as store_name, i.name as item_name, l.id as item_weight " +
                "FROM ge_line_item l, ge_order o, ge_item i " +
                "WHERE l.order_id = (SELECT id FROM ge_order WHERE order_id = '" + orderID + "') " +
                "AND o.store_id = (SELECT id FROM ge_store WHERE name = '" + storeName + "') " +
                "AND l.order_id = o.id " +
                "AND i.store_id = (SELECT id FROM ge_store WHERE name = '" + storeName + "') " +
                "AND l.item_id = i.id;",itemRowMapper);
    }

    /**
     * @param storeName
     * @param orderId
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public List<OrderItem> getItemsByOrder(String storeName, String orderId) throws SQLException, ClassNotFoundException {
        return  orderItemConnectionManager.getEntities(
                "SELECT i.name AS item_name, li.quantity AS total_quantity, " +
                        "(li.quantity * li.unit_price) AS total_cost, (li.quantity * i.weight) AS total_weight " +
                        "FROM ge_line_item li " +
                        "INNER JOIN ge_item i ON li.item_id = i.id " +
                        "INNER JOIN ge_order o ON o.id = li.order_id " +
                        "INNER JOIN ge_store s ON s.id = o.store_id " +
                        "WHERE s.name='" + storeName + "' " +
                        "AND o.order_id='" + orderId + "' " +
                        "ORDER BY item_name ASC;",
                orderItemRowMapper
        );
    }
}
