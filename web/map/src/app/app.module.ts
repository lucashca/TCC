import { BrowserModule } from "@angular/platform-browser";
import { NgModule } from "@angular/core";

import { AppRoutingModule } from "./app-routing.module";
import { AppComponent } from "./app.component";
import { BrowserAnimationsModule } from "@angular/platform-browser/animations";
import { MapComponent } from "./map/map.component";
import { AgmCoreModule } from "@agm/core";
import { HttpClientModule } from "@angular/common/http";

@NgModule({
  declarations: [AppComponent, MapComponent],
  imports: [
    BrowserModule,
    HttpClientModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    AgmCoreModule.forRoot({
      apiKey: "AIzaSyBPVcAtUxeYITEYghEnWvg_4_OmTsME_g0"
    })
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
