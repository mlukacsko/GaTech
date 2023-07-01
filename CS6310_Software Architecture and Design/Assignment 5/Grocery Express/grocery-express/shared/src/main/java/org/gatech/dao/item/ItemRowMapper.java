package org.gatech.dao.item;

import org.gatech.dto.Item;
import org.gatech.dbconnect.RowMapper;

import java.sql.ResultSet;
import java.sql.SQLException;

public class ItemRowMapper implements RowMapper<Item> {
    @Override
    public Item mapRow(ResultSet rs) throws SQLException {
        return new Item.ItemBuilder()
                .withStoreName(rs.getString("store_name"))
                .withItemName(rs.getString("item_name"))
                .withItemWeight(rs.getInt("item_weight"))
                .build();
    }
}
