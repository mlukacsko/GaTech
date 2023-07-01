package groceryexpress;

import com.google.inject.AbstractModule;
import com.google.inject.TypeLiteral;
import org.gatech.dao.coupon.CouponDao;
import org.gatech.dao.coupon.DefaultCouponDao;
import org.gatech.dao.drone.DefaultDroneDao;
import org.gatech.dao.drone.DroneDao;
import org.gatech.dao.customer.CustomerDao;
import org.gatech.dao.customer.DefaultCustomerDao;
import org.gatech.dao.item.DefaultItemDao;
import org.gatech.dao.item.ItemDao;
import org.gatech.dao.location.DefaultLocationDao;
import org.gatech.dao.location.LocationDao;
import org.gatech.dao.order.DefaultOrderDao;
import org.gatech.dao.order.OrderDao;
import org.gatech.dao.pilot.DefaultDronePilotDao;
import org.gatech.dao.pilot.DronePilotDao;
import org.gatech.dao.store.DefaultStoreDao;
import org.gatech.dao.store.StoreDao;
import org.gatech.dao.settings.DefaultSettingsDao;
import org.gatech.dao.settings.SettingsDao;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dbconnect.DataConnectionManager;
import org.gatech.dto.*;
import org.gatech.service.customer.CustomerService;
import org.gatech.service.customer.DefaultCustomerService;
import org.gatech.service.location.DefaultLocationService;
import org.gatech.service.location.LocationService;
import org.gatech.service.settings.DefaultSettingsService;
import org.gatech.service.settings.SettingsService;
import org.gatech.service.pilot.DefaultDronePilotService;
import org.gatech.service.pilot.DronePilotService;
import org.gatech.service.store.DefaultStoreService;
import org.gatech.service.store.StoreService;


public class ApplicationModule extends AbstractModule {

    @Override
    protected void configure() {

        // services
        bind(StoreService.class).to(DefaultStoreService.class);
        bind(CustomerService.class).to(DefaultCustomerService.class);
        bind(DronePilotService.class).to(DefaultDronePilotService.class);
        bind(SettingsService.class).to(DefaultSettingsService.class);
        bind(LocationService.class).to(DefaultLocationService.class);

        // dao
        bind(StoreDao.class).to(DefaultStoreDao.class);
        bind(ItemDao.class).to(DefaultItemDao.class);
        bind(CustomerDao.class).to(DefaultCustomerDao.class);
        bind(DronePilotDao.class).to(DefaultDronePilotDao.class);
        bind(DroneDao.class).to(DefaultDroneDao.class);
        bind(OrderDao.class).to(DefaultOrderDao.class);
        bind(CouponDao.class).to(DefaultCouponDao.class);
        bind(SettingsDao.class).to(DefaultSettingsDao.class);
        bind(LocationDao.class).to(DefaultLocationDao.class);

        // connection manager
        bind(new TypeLiteral<ConnectionManager<Store>>(){}).toInstance(new DataConnectionManager<>( "org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Item>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<OrderItem>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Customer>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<DronePilot>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Drone>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Order>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Coupon>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Settings>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
        bind(new TypeLiteral<ConnectionManager<Location>>(){}).toInstance(new DataConnectionManager<>("org.postgresql.Driver"));
    }
}
