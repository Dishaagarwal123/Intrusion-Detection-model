let length = 0;
setInterval(() => {
  if (length < data.length) {
    for (; length < data.length; length++) {
      const newdiv = document.createElement("div");
      newdiv.innerHTML =
        "<p><b>IP address : </b> <%= data[length].source_address_ip %></p><p><b>Ethernet address: </b> <%= data[length].src_eth_add %></p><p><b>Source Port : </b><%= data[length].source_port %></p>";
      document.body.prepend(newdiv);
    }
  }
}, 500);
