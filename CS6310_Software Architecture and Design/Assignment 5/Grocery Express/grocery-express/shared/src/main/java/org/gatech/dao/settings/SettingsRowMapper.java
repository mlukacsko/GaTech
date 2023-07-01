package org.gatech.dao.settings;

import org.gatech.dbconnect.RowMapper;
import org.gatech.dto.Settings;

import java.sql.ResultSet;
import java.sql.SQLException;

public class SettingsRowMapper implements RowMapper<Settings> {
    @Override
    public Settings mapRow(ResultSet rs) throws SQLException {
        return new Settings.SettingsBuilder()
                .withTime(rs.getString("clock"))
                .withDroneEnergyConsumption(rs.getInt("drone_energy_consumption"))
                .withDroneEnergyCapacity(rs.getInt("drone_energy_capacity"))
                .withDroneEnergyRestorationT0000(rs.getFloat("t_0000"))
                .withDroneEnergyRestorationT0100(rs.getFloat("t_0100"))
                .withDroneEnergyRestorationT0200(rs.getFloat("t_0200"))
                .withDroneEnergyRestorationT0300(rs.getFloat("t_0300"))
                .withDroneEnergyRestorationT0400(rs.getFloat("t_0400"))
                .withDroneEnergyRestorationT0500(rs.getFloat("t_0500"))
                .withDroneEnergyRestorationT0600(rs.getFloat("t_0600"))
                .withDroneEnergyRestorationT0700(rs.getFloat("t_0700"))
                .withDroneEnergyRestorationT0800(rs.getFloat("t_0800"))
                .withDroneEnergyRestorationT0900(rs.getFloat("t_0900"))
                .withDroneEnergyRestorationT1000(rs.getFloat("t_1000"))
                .withDroneEnergyRestorationT1100(rs.getFloat("t_1100"))
                .withDroneEnergyRestorationT1200(rs.getFloat("t_1200"))
                .withDroneEnergyRestorationT1300(rs.getFloat("t_1300"))
                .withDroneEnergyRestorationT1400(rs.getFloat("t_1400"))
                .withDroneEnergyRestorationT1500(rs.getFloat("t_1500"))
                .withDroneEnergyRestorationT1600(rs.getFloat("t_1600"))
                .withDroneEnergyRestorationT1700(rs.getFloat("t_1700"))
                .withDroneEnergyRestorationT1800(rs.getFloat("t_1800"))
                .withDroneEnergyRestorationT1900(rs.getFloat("t_1900"))
                .withDroneEnergyRestorationT2000(rs.getFloat("t_2000"))
                .withDroneEnergyRestorationT2100(rs.getFloat("t_2100"))
                .withDroneEnergyRestorationT2200(rs.getFloat("t_2200"))
                .withDroneEnergyRestorationT2300(rs.getFloat("t_2300"))
                .build();
    }
}
