package groceryexpress;

import com.google.inject.Guice;
import com.google.inject.Injector;

public class App {
    public static void main(String[] args) {
        System.out.println("Welcome to the Grocery Express Delivery Service!");
        Injector injector = Guice.createInjector(new ApplicationModule());
        DeliveryService deliveryService = injector.getInstance(DeliveryService.class);
        CLI cli = new CLI(deliveryService);
        cli.commandLoop();
    }
}
