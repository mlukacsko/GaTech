package org.gatech.dbconnect;

import java.sql.SQLException;
import java.util.List;

public interface ConnectionManager<T extends DatabaseEntity> {

    /**
     * Retrieve a list of entities from the database
     * @param sql - query to be executed
     * @param rowMapper - org.jfoshee.db.util.RowMapper instance to map the result set to a dto
     * @return a list of mapped objects from the database
     * @throws SQLException is thrown if the query is invalid
     * @throws ClassNotFoundException is thrown if
     */
    public List<T> getEntities(String sql, RowMapper<T> rowMapper) throws SQLException, ClassNotFoundException;

    /**
     * Retrieve single entity from the database
     * @param sql - query to be executed
     * @param rowMapper - org.jfoshee.db.util.RowMapper instance to map the result set to a dto
     * @return a single mapped object from the database
     * @throws SQLException is thrown if the query is invalid
     * @throws ClassNotFoundException is thrown if
     */
    public T getSingleEntity(String sql, RowMapper<T> rowMapper) throws SQLException, ClassNotFoundException;

    /**
     * Execute SQL statements that are not supposed to return a result set (i.e. INSERT, UPDATE, DELETE)
     * @param sql - sql statement to be executed
     * @throws SQLException is thrown if the query is invalid
     * @throws ClassNotFoundException is thrown if
     */
    public void executeSql(String sql) throws SQLException, ClassNotFoundException;

    /**
     * Execute SQL statements that is supposed to return a result
     * @param sql
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    public int executeSqlWithReturn(String sql) throws SQLException, ClassNotFoundException;

}
