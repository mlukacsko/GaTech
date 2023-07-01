package org.gatech.dbconnect;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class DataConnectionManager<T extends DatabaseEntity> implements ConnectionManager<T> {

    private final String DEFAULT_CONNECTION_HOST = "localhost";
    private final String DEFAULT_CONNECTION_PORT = "15432";
    private final String DEFAULT_CONNECTION_DB = "postgres";
    private final String DEFAULT_CONNECTION_USER = "postgres";
    private final String DEFAULT_CONNECTION_PASSWORD = "postgres";

    private String dbHost;
    private String dbPort;
    private String dbDatabase;
    private String dbUser;
    private String dbPassword;

    private String driverClass;

    public DataConnectionManager(String driverClass) {
        this.driverClass = driverClass;

        String host = System.getenv("POSTGRES_HOST");
        String port = System.getenv("POSTGRES_PORT");
        String db = System.getenv("POSTGRES_DB");
        String user = System.getenv("POSTGRES_USER");
        String password = System.getenv("POSTGRES_PASSWORD");

        dbHost = host != null ? host : DEFAULT_CONNECTION_HOST;
        dbPort = port != null ? port : DEFAULT_CONNECTION_PORT;
        dbDatabase = db != null ? db : DEFAULT_CONNECTION_DB;
        dbUser = user != null ? user : DEFAULT_CONNECTION_USER;
        dbPassword = password != null ? password : DEFAULT_CONNECTION_PASSWORD;
    }

    private String getConnectionUrl() {
        return "jdbc:postgresql://" + dbHost +
                ":" + dbPort +
                "/" + dbDatabase +
                "?user=" + dbUser +
                "&password=" + dbPassword;
    }

    @Override
    public List<T> getEntities(String sql, RowMapper<T> rowMapper) throws SQLException, ClassNotFoundException {
        Class.forName(driverClass);

        String url = getConnectionUrl();

        Connection connection = DriverManager.getConnection(url);
        Statement statement = connection.createStatement();

        ResultSet rs = statement.executeQuery(sql);
        List<T> entities = new ArrayList<>();
        while (rs.next()) {
            entities.add(rowMapper.mapRow(rs));
        }

        connection.close();
        statement.close();

        return entities;
    }

    @Override
    public T getSingleEntity(String sql, RowMapper<T> rowMapper) throws SQLException, ClassNotFoundException {
        Class.forName(driverClass);

        String url = getConnectionUrl();

        Connection connection = DriverManager.getConnection(url);
        Statement statement = connection.createStatement();

        ResultSet rs = statement.executeQuery(sql);

        T result = rowMapper.mapRow(rs);

        connection.close();
        statement.close();
        return result;
    }

    @Override
    public void executeSql(String sql) throws SQLException, ClassNotFoundException {
        Class.forName(driverClass);

        String url = getConnectionUrl();

        Connection conn = DriverManager.getConnection(url);
        Statement statement = conn.createStatement();

        statement.executeUpdate(sql);

        conn.close();
        statement.close();
    }

    /**
     * Execute SQL statements that is supposed to return a result
     *
     * @param sql
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public int executeSqlWithReturn(String sql) throws SQLException, ClassNotFoundException {
        Class.forName(driverClass);

        String url = getConnectionUrl();

        Connection connection = DriverManager.getConnection(url);
        Statement statement = connection.createStatement();

        ResultSet rs = statement.executeQuery(sql);

        // purpose of this is to get the id of the last inserted record
        rs.next();  // need to call next before you get the columns
        int result = rs.getInt("id");

        connection.close();
        statement.close();

        return result;
    }
}

