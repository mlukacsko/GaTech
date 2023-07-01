package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class Item extends DatabaseEntity {

    private String storeName;
    private String itemName;
    private int itemWeight;

    public Item(String storeName, String itemName, int itemWeight) {
        this.storeName = storeName;
        this.itemName = itemName;
        this.itemWeight = itemWeight;
    }

    private Item(ItemBuilder builder) {
        this.storeName = builder.storeName;
        this.itemName = builder.itemName;
        this.itemWeight = builder.itemWeight;
    }

    public String getName() {
        return storeName;
    }

    public void setName(String name) {
        this.storeName = name;
    }

    public int getItemWeight() {
        return itemWeight;
    }

    public void setItemWeight(int itemWeight) {
        this.itemWeight = itemWeight;
    }

    public String getItemName() {
        return itemName;
    }

    public void setItemName(String itemName) {
        this.itemName = itemName;
    }

    @Override
    public String toString() {
        return itemName + "," + itemWeight;
    }

    public static class ItemBuilder {
        private String storeName;
        private String itemName;
        private int itemWeight;

        public Item build() {
            return new Item(this);
        }

        public ItemBuilder withStoreName(String storeName) {
            this.storeName = storeName;
            return this;
        }

        public ItemBuilder withItemName(String itemName) {
            this.itemName = itemName;
            return this;
        }

        public ItemBuilder withItemWeight(int itemWeight) {
            this.itemWeight = itemWeight;
            return this;
        }

    }
}
