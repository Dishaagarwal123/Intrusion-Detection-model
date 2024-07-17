import express from "express";
import bodyParser from "body-parser";
import { dirname } from "path";

const port = 8000;
const pwd = dirname(process.argv[1]);

const app = express();

app.use(express.static(pwd + "/public"));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: true }));

let data = [];
let color = {
  NORMAL: "white",
  ARP_POISONING: "red",
  WARM_RESTART: "red",
  COLD_RESTART: "red",
  MITM_DOS: "red",
  DISABLE_UNSOLICITED: "yellow",
  REPLAY: "yellow",
  DNP3_ENUMERATE: "yellow",
  DNP3_INFO: "yellow",
  INIT_DATA: "red",
  STOP_APP: "red",
};

app.get("/", (req, res) => {
  res.render("index.ejs", { data, color });
});

app.get("/packet/:id", (req, res) => {
  let { id } = req.params;
  res.render("packet.ejs", { packet: data[id] });
});

app.post("/submit", (req, res) => {
  data.push(...JSON.parse(req.body.key));
  console.log(data);
  res.sendStatus(200);
});

app.listen(port, () => {
  console.log("Server running on port : ", port);
});
