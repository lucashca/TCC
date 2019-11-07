import { Injectable } from "@angular/core";

import { Coordinates } from "../class/coordinates";

import { cords } from "./markers";

import "./markers.js";
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { TouchSequence } from 'selenium-webdriver';

@Injectable({
  providedIn: "root"
})
export class MapService {
  markers: Coordinates[] = [];

  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
    })
  }

  constructor(public http: HttpClient) { }


  setMarkers() {
    for (let c of cords) {
      this.markers.push(new Coordinates(c[0], c[1]));
    }
  }

  getMarkers() {
    this.setMarkers();

    return this.http.get("assets/dataset.json");
  }

  getElevation(location) {
    return this.http.get('https://api.jawg.io/elevations?locations=' + location + '&access-token=I0tZmMAxflhrsGdj0CfDBZEwFljBzlqcVQZz6cQPFfVecsCkmdGfi0odUi3HS7iD', this.httpOptions)
  }

  getNutrienteData(lat, lng, elevations, ca) {
    let url = "http://localhost:8080";
    return this.http.post(url, { latitude: lat, longitude: lng, elevation: elevations, ca: ca })
  }

}
