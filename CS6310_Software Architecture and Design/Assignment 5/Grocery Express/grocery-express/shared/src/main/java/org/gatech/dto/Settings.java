package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class Settings extends DatabaseEntity {
    private String clock;
    private int droneEnergyConsumption;
    private int droneEnergyCapacity;

    public float droneEnergyRestorationT0000;
    public float droneEnergyRestorationT0100;
    public float droneEnergyRestorationT0200;
    public float droneEnergyRestorationT0300;
    public float droneEnergyRestorationT0400;
    public float droneEnergyRestorationT0500;
    public float droneEnergyRestorationT0600;
    public float droneEnergyRestorationT0700;
    public float droneEnergyRestorationT0800;
    public float droneEnergyRestorationT0900;
    public float droneEnergyRestorationT1000;
    public float droneEnergyRestorationT1100;
    public float droneEnergyRestorationT1200;
    public float droneEnergyRestorationT1300;
    public float droneEnergyRestorationT1400;
    public float droneEnergyRestorationT1500;
    public float droneEnergyRestorationT1600;
    public float droneEnergyRestorationT1700;
    public float droneEnergyRestorationT1800;
    public float droneEnergyRestorationT1900;
    public float droneEnergyRestorationT2000;
    public float droneEnergyRestorationT2100;
    public float droneEnergyRestorationT2200;
    public float droneEnergyRestorationT2300;

    public Settings(
            String clock,
            int droneEnergyConsumption,
            int droneEnergyCapacity,
            float droneEnergyRestorationT0000,
            float droneEnergyRestorationT0100,
            float droneEnergyRestorationT0200,
            float droneEnergyRestorationT0300,
            float droneEnergyRestorationT0400,
            float droneEnergyRestorationT0500,
            float droneEnergyRestorationT0600,
            float droneEnergyRestorationT0700,
            float droneEnergyRestorationT0800,
            float droneEnergyRestorationT0900,
            float droneEnergyRestorationT1000,
            float droneEnergyRestorationT1100,
            float droneEnergyRestorationT1200,
            float droneEnergyRestorationT1300,
            float droneEnergyRestorationT1400,
            float droneEnergyRestorationT1500,
            float droneEnergyRestorationT1600,
            float droneEnergyRestorationT1700,
            float droneEnergyRestorationT1800,
            float droneEnergyRestorationT1900,
            float droneEnergyRestorationT2000,
            float droneEnergyRestorationT2100,
            float droneEnergyRestorationT2200,
            float droneEnergyRestorationT2300
    ) {
        this.clock = clock;
        this.droneEnergyConsumption = droneEnergyConsumption;
        this.droneEnergyCapacity = droneEnergyCapacity;
        this.droneEnergyRestorationT0000 = droneEnergyRestorationT0000;
        this.droneEnergyRestorationT0100 = droneEnergyRestorationT0100;
        this.droneEnergyRestorationT0200 = droneEnergyRestorationT0200;
        this.droneEnergyRestorationT0300 = droneEnergyRestorationT0300;
        this.droneEnergyRestorationT0400 = droneEnergyRestorationT0400;
        this.droneEnergyRestorationT0500 = droneEnergyRestorationT0500;
        this.droneEnergyRestorationT0600 = droneEnergyRestorationT0600;
        this.droneEnergyRestorationT0700 = droneEnergyRestorationT0700;
        this.droneEnergyRestorationT0800 = droneEnergyRestorationT0800;
        this.droneEnergyRestorationT0900 = droneEnergyRestorationT0900;
        this.droneEnergyRestorationT1000 = droneEnergyRestorationT1000;
        this.droneEnergyRestorationT1100 = droneEnergyRestorationT1100;
        this.droneEnergyRestorationT1200 = droneEnergyRestorationT1200;
        this.droneEnergyRestorationT1300 = droneEnergyRestorationT1300;
        this.droneEnergyRestorationT1400 = droneEnergyRestorationT1400;
        this.droneEnergyRestorationT1500 = droneEnergyRestorationT1500;
        this.droneEnergyRestorationT1600 = droneEnergyRestorationT1600;
        this.droneEnergyRestorationT1700 = droneEnergyRestorationT1700;
        this.droneEnergyRestorationT1800 = droneEnergyRestorationT1800;
        this.droneEnergyRestorationT1900 = droneEnergyRestorationT1900;
        this.droneEnergyRestorationT2000 = droneEnergyRestorationT2000;
        this.droneEnergyRestorationT2100 = droneEnergyRestorationT2100;
        this.droneEnergyRestorationT2200 = droneEnergyRestorationT2200;
        this.droneEnergyRestorationT2300 = droneEnergyRestorationT2300;
    }

    private Settings(SettingsBuilder settingsBuilder) {
        this.clock = settingsBuilder.clock;
        this.droneEnergyConsumption = settingsBuilder.droneEnergyConsumption;
        this.droneEnergyCapacity = settingsBuilder.droneEnergyCapacity;
        this.droneEnergyRestorationT0000 = settingsBuilder.droneEnergyRestorationT0000;
        this.droneEnergyRestorationT0100 = settingsBuilder.droneEnergyRestorationT0100;
        this.droneEnergyRestorationT0200 = settingsBuilder.droneEnergyRestorationT0200;
        this.droneEnergyRestorationT0300 = settingsBuilder.droneEnergyRestorationT0300;
        this.droneEnergyRestorationT0400 = settingsBuilder.droneEnergyRestorationT0400;
        this.droneEnergyRestorationT0500 = settingsBuilder.droneEnergyRestorationT0500;
        this.droneEnergyRestorationT0600 = settingsBuilder.droneEnergyRestorationT0600;
        this.droneEnergyRestorationT0700 = settingsBuilder.droneEnergyRestorationT0700;
        this.droneEnergyRestorationT0800 = settingsBuilder.droneEnergyRestorationT0800;
        this.droneEnergyRestorationT0900 = settingsBuilder.droneEnergyRestorationT0900;
        this.droneEnergyRestorationT1000 = settingsBuilder.droneEnergyRestorationT1000;
        this.droneEnergyRestorationT1100 = settingsBuilder.droneEnergyRestorationT1100;
        this.droneEnergyRestorationT1200 = settingsBuilder.droneEnergyRestorationT1200;
        this.droneEnergyRestorationT1300 = settingsBuilder.droneEnergyRestorationT1300;
        this.droneEnergyRestorationT1400 = settingsBuilder.droneEnergyRestorationT1400;
        this.droneEnergyRestorationT1500 = settingsBuilder.droneEnergyRestorationT1500;
        this.droneEnergyRestorationT1600 = settingsBuilder.droneEnergyRestorationT1600;
        this.droneEnergyRestorationT1700 = settingsBuilder.droneEnergyRestorationT1700;
        this.droneEnergyRestorationT1800 = settingsBuilder.droneEnergyRestorationT1800;
        this.droneEnergyRestorationT1900 = settingsBuilder.droneEnergyRestorationT1900;
        this.droneEnergyRestorationT2000 = settingsBuilder.droneEnergyRestorationT2000;
        this.droneEnergyRestorationT2100 = settingsBuilder.droneEnergyRestorationT2100;
        this.droneEnergyRestorationT2200 = settingsBuilder.droneEnergyRestorationT2200;
        this.droneEnergyRestorationT2300 = settingsBuilder.droneEnergyRestorationT2300;
    }

    public String getClock() {
        return clock;
    }
    public int getDroneEnergyConsumption() { return droneEnergyConsumption; }
    public int getDroneEnergyCapacity() { return droneEnergyCapacity; }

    public void setClock(String clock) {
        this.clock = clock;
    }
    public void setDroneEnergyConsumption(int droneEnergyConsumption) {
        this.droneEnergyConsumption = droneEnergyConsumption;
    }
    public void setDroneEnergyCapacity(int droneEnergyCapacity) {
        this.droneEnergyCapacity = droneEnergyCapacity;
    }

    public String toString() {
        return "clock:" + clock + ",energy_consumption:" + droneEnergyConsumption + ",energy_capacity:" +
                droneEnergyCapacity +
                ",energy_restoration_t0000:" + droneEnergyRestorationT0000 +
                ",energy_restoration_t0100:" + droneEnergyRestorationT0100 +
                ",energy_restoration_t0200:" + droneEnergyRestorationT0200 +
                ",energy_restoration_t0300:" + droneEnergyRestorationT0300 +
                ",energy_restoration_t0400:" + droneEnergyRestorationT0400 +
                ",energy_restoration_t0500:" + droneEnergyRestorationT0500 +
                ",energy_restoration_t0600:" + droneEnergyRestorationT0600 +
                ",energy_restoration_t0700:" + droneEnergyRestorationT0700 +
                ",energy_restoration_t0800:" + droneEnergyRestorationT0800 +
                ",energy_restoration_t0900:" + droneEnergyRestorationT0900 +
                ",energy_restoration_t1000:" + droneEnergyRestorationT1000 +
                ",energy_restoration_t1100:" + droneEnergyRestorationT1100 +
                ",energy_restoration_t1200:" + droneEnergyRestorationT1200 +
                ",energy_restoration_t1300:" + droneEnergyRestorationT1300 +
                ",energy_restoration_t1400:" + droneEnergyRestorationT1400 +
                ",energy_restoration_t1500:" + droneEnergyRestorationT1500 +
                ",energy_restoration_t1600:" + droneEnergyRestorationT1600 +
                ",energy_restoration_t1700:" + droneEnergyRestorationT1700 +
                ",energy_restoration_t1800:" + droneEnergyRestorationT1800 +
                ",energy_restoration_t1900:" + droneEnergyRestorationT1900 +
                ",energy_restoration_t2000:" + droneEnergyRestorationT2000 +
                ",energy_restoration_t2100:" + droneEnergyRestorationT2100 +
                ",energy_restoration_t2200:" + droneEnergyRestorationT2200 +
                ",energy_restoration_t2300:" + droneEnergyRestorationT2300;
    }

    public static class SettingsBuilder {
        private String clock;
        private int droneEnergyConsumption;
        private int droneEnergyCapacity;

        private float droneEnergyRestorationT0000;
        private float droneEnergyRestorationT0100;
        private float droneEnergyRestorationT0200;
        private float droneEnergyRestorationT0300;
        private float droneEnergyRestorationT0400;
        private float droneEnergyRestorationT0500;
        private float droneEnergyRestorationT0600;
        private float droneEnergyRestorationT0700;
        private float droneEnergyRestorationT0800;
        private float droneEnergyRestorationT0900;
        private float droneEnergyRestorationT1000;
        private float droneEnergyRestorationT1100;
        private float droneEnergyRestorationT1200;
        private float droneEnergyRestorationT1300;
        private float droneEnergyRestorationT1400;
        private float droneEnergyRestorationT1500;
        private float droneEnergyRestorationT1600;
        private float droneEnergyRestorationT1700;
        private float droneEnergyRestorationT1800;
        private float droneEnergyRestorationT1900;
        private float droneEnergyRestorationT2000;
        private float droneEnergyRestorationT2100;
        private float droneEnergyRestorationT2200;
        private float droneEnergyRestorationT2300;

        public Settings build() {
            return new Settings(this);
        }

        public SettingsBuilder withTime(String clock) {
            this.clock = clock;
            return this;
        }

        public SettingsBuilder withDroneEnergyConsumption(int droneEnergyConsumption) {
            this.droneEnergyConsumption = droneEnergyConsumption;
            return this;
        }

        public SettingsBuilder withDroneEnergyCapacity(int droneEnergyCapacity) {
            this.droneEnergyCapacity = droneEnergyCapacity;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0000(float droneEnergyRestorationT0000) {
            this.droneEnergyRestorationT0000 = droneEnergyRestorationT0000;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0100(float droneEnergyRestorationT0100) {
            this.droneEnergyRestorationT0100 = droneEnergyRestorationT0100;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0200(float droneEnergyRestorationT0200) {
            this.droneEnergyRestorationT0200 = droneEnergyRestorationT0200;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0300(float droneEnergyRestorationT0300) {
            this.droneEnergyRestorationT0300 = droneEnergyRestorationT0300;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0400(float droneEnergyRestorationT0400) {
            this.droneEnergyRestorationT0400 = droneEnergyRestorationT0400;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0500(float droneEnergyRestorationT0500) {
            this.droneEnergyRestorationT0500 = droneEnergyRestorationT0500;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0600(float droneEnergyRestorationT0600) {
            this.droneEnergyRestorationT0600 = droneEnergyRestorationT0600;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0700(float droneEnergyRestorationT0700) {
            this.droneEnergyRestorationT0700 = droneEnergyRestorationT0700;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0800(float droneEnergyRestorationT0800) {
            this.droneEnergyRestorationT0800 = droneEnergyRestorationT0800;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT0900(float droneEnergyRestorationT0900) {
            this.droneEnergyRestorationT0900 = droneEnergyRestorationT0900;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1000(float droneEnergyRestorationT1000) {
            this.droneEnergyRestorationT1000 = droneEnergyRestorationT1000;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1100(float droneEnergyRestorationT1100) {
            this.droneEnergyRestorationT1100 = droneEnergyRestorationT1100;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1200(float droneEnergyRestorationT1200) {
            this.droneEnergyRestorationT1200 = droneEnergyRestorationT1200;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1300(float droneEnergyRestorationT1300) {
            this.droneEnergyRestorationT1300 = droneEnergyRestorationT1300;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1400(float droneEnergyRestorationT1400) {
            this.droneEnergyRestorationT1400 = droneEnergyRestorationT1400;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1500(float droneEnergyRestorationT1500) {
            this.droneEnergyRestorationT1500 = droneEnergyRestorationT1500;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1600(float droneEnergyRestorationT1600) {
            this.droneEnergyRestorationT1600 = droneEnergyRestorationT1600;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1700(float droneEnergyRestorationT1700) {
            this.droneEnergyRestorationT1700 = droneEnergyRestorationT1700;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1800(float droneEnergyRestorationT1800) {
            this.droneEnergyRestorationT1800 = droneEnergyRestorationT1800;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT1900(float droneEnergyRestorationT1900) {
            this.droneEnergyRestorationT1900 = droneEnergyRestorationT1900;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT2000(float droneEnergyRestorationT2000) {
            this.droneEnergyRestorationT2000 = droneEnergyRestorationT2000;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT2100(float droneEnergyRestorationT2100) {
            this.droneEnergyRestorationT2100 = droneEnergyRestorationT2100;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT2200(float droneEnergyRestorationT2200) {
            this.droneEnergyRestorationT2200 = droneEnergyRestorationT2200;
            return this;
        }

        public SettingsBuilder withDroneEnergyRestorationT2300(float droneEnergyRestorationT2300) {
            this.droneEnergyRestorationT2300 = droneEnergyRestorationT2300;
            return this;
        }
    }
}
