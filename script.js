CUR_STATE = null;
CUR_STATE_REWARD = 0;
ACTION_TAKEN = null;

function Ai() {

    this.init = function() {
        $.post('http://192.168.0.104:5000/api/initialize', function( data ) {});
    }

    this.restart = function() {
        $.ajax({
          type: "POST",
          contentType: "application/json; charset=utf-8",
          url: "http://192.168.0.104:5000/api/restart",
          data: JSON.stringify({"state": CUR_STATE, "next_state": CUR_STATE, "reward": -CUR_STATE_REWARD, "action_taken": ACTION_TAKEN}),
          success: function( data ) {},
          dataType: "json"
        });
        document.getElementsByClassName("retry-button")[0].click();
    }

    this.step = function(grid) {
        if ($("div.game-message.game-over p").text() != "") {
            console.log("h")
            this.restart();
            return 0;
        }
        // This method is called on every update.
        // Return one of these integers to move tiles on the grid:
        // 0: up, 1: right, 2: down, 3: left

        // Parameter grid contains current state of the game as Tile objects stored in grid.cells.
        // Top left corner is at grid.cells[0][0], top right: grid.cells[3][0], bottom left: grid.cells[0][3], bottom right: grid.cells[3][3].
        // Tile objects have .value property which contains the value of the tile. If top left corner has tile with 2, grid.cells[0][0].value == 2.
        // Array will contain null if there is no tile in the slot (e.g. grid.cells[0][3] == null if bottom left corner doesn't have a tile).

        // Grid has 2 useful helper methods:
        // .copy()    - creates a copy of the grid and returns it.
        // .move(dir) - can be used to determine what is the next state of the grid going to be if moved to that direction.
        //              This changes the state of the grid object, so you should probably copy() the grid before using this.
        //              Naturally the modified state doesn't contain information about new tiles.
        //              Method returns true if you can move to that direction, false otherwise.

        // sample AI:
        var state = {};
        var reward = 0;
        var cloned = grid.copy();
        for (var i = 0; i < cloned.cells.length; i++) {
            for (var j = 0; j < cloned.cells[i].length; j++) {
                cell = cloned.cells[i][j];
                cell_name = "" + i + "_" + j
                state[cell_name] = cell != null ? cell.value : 0
                reward += cell != null ? cell.value : 0
            }
        }

        if (CUR_STATE != null) {
            $.ajax({
              type: "POST",
              contentType: "application/json; charset=utf-8",
              url: "http://192.168.0.104:5000/api/reward_update",
              data: JSON.stringify({"state": CUR_STATE, "next_state": CUR_STATE, "reward": reward, "action_taken": ACTION_TAKEN}),
              success: function( data ) {},
              dataType: "json"
            });
        }

        CUR_STATE = state;
        CUR_STATE_REWARD = reward

        var illegals = [];
        for(var i=0; i<4; i+=1) {
            cloned = grid.copy();
            if(!cloned.move(i)) {
                illegals.push(i);
            }
        }

        var action;

        $.ajax({
              async: false,
              type: "POST",
              contentType: "application/json; charset=utf-8",
              url: "http://192.168.0.104:5000/api/get_action",
              data: JSON.stringify({"state": CUR_STATE, "illegals": illegals}),
              success: function( data ) {
                action = data["action"]
              },
              dataType: "json"
            });
        ACTION_TAKEN = action
        return action
    }
}

function compute_reward(state) {


}