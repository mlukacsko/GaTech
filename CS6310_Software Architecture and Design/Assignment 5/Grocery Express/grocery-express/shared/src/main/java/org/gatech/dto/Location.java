package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class Location extends DatabaseEntity {
    private int x;
    private int y;

    public Location(int x, int y) {
        this.x = x;
        this.y = y;
    }

    private Location(LocationBuilder builder) {
        this.x = builder.x;
        this.y = builder.y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public void setX(int x) {
        this.x = x;
    }

    public void setY(int y) {
        this.y = y;
    }

    public String toString() {
        return "x:"+this.x+",y:"+this.y;
    }

    public static class LocationBuilder {
        private int x;
        private int y;

        public Location build() {
            return new Location(this);
        }

        public LocationBuilder withX(int x) {
            this.x = x;
            return this;
        }

        public LocationBuilder withY(int y) {
            this.y = y;
            return this;
        }
    }
}
