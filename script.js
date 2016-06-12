CUR_STATE = null;
NEXT_STATE = null;

function Ai() {

    this.init = function() {
        $.ajax({
            url:'/api/initialize',
            type:'post',
            success:function(){}
    });
    }

    this.restart = function() {
        $.ajax({
            url:'/api/restart',
            type:'post',
            success:function(){}
    }

    this.step = function(grid) {
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
        console.log(grid)

        if (this.foo == null) this.foo = 0;
        return this.foo++ % 4;
    }
}